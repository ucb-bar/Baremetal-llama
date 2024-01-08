/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

//#include "weights.h"
//#include "tokenizer.h"
#include "2M_weights.h"
#include "tok4096.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */


// ----------------------------------------------------------------------------
// Globals
float KV_CACHE_SCALE = 8.0f/127.0f; // magic number for now... QuantizeTensor errors
size_t SPARSECOUNT = 0; // track how many horrifying matmuls we saved from activation
// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    int8_t* q;  // quantized values
    float s; // scaling factor
} QuantizedTensor;

typedef struct {
    // token embedding table
    QuantizedTensor *q_tokens; // (vocab_size, dim)

    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    QuantizedTensor *wq; // (layer, dim, n_heads * head_size)
    QuantizedTensor *wk; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wv; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor *wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    QuantizedTensor *w1; // (layer, hidden_dim, dim)
    QuantizedTensor *w2; // (layer, dim, hidden_dim)
    QuantizedTensor *w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    int8_t* key_cache; //QuantizedTensor key_cache; //float* key_cache;   // (layer, seq_len, dim)
    int8_t* value_cache; //QuantizedTensor value_cache; //float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->xq = (QuantizedTensor) { .q = calloc(p->dim, sizeof(int8_t)), .s = 0.0f };
    s->hq = (QuantizedTensor) { .q = calloc(p->hidden_dim, sizeof(int8_t)), .s = 0.0f };
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(kv_dim, sizeof(float));
    s->v = calloc(kv_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));

    int kv_cache_dim = p->n_layers * p->seq_len * kv_dim;
    //s->key_cache = (QuantizedTensor) { .q = calloc(kv_cache_dim, sizeof(int8_t)), .s = calloc(kv_cache_dim/group_size, sizeof(float)) };
    //s->value_cache = (QuantizedTensor) { .q = calloc(kv_cache_dim, sizeof(int8_t)), .s = calloc(kv_cache_dim/group_size, sizeof(float)) };
    s->key_cache = calloc(kv_cache_dim, sizeof(int8_t));
    s->value_cache = calloc(kv_cache_dim, sizeof(int8_t));

    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->k || !s->v || !s->att || !s->logits 
     || !s->xq.q || !s->hq.q 
     //|| !s->key_cache.q || !s->key_cache.s || !s->value_cache.q || !s->value_cache.s
     || !s->key_cache || !s->value_cache
     ) {
        printf("malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->xb2);
    free(s->hb);
    free(s->hb2);
    free(s->xq.q);
    free(s->hq.q);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);

    free(s->key_cache);
    free(s->value_cache);
    /*
    free(s->key_cache.q);
    free(s->key_cache.s);
    free(s->value_cache.q);
    free(s->value_cache.s);
    */
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, float* x, int n, int offset) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[offset+i] * qx->s;
    }
}

void quantize(QuantizedTensor *qx, float* x, int n) {
    float Q_MAX = 127.0f;
    // find the max absolute value in the current group
    float wmax = 0.0;
    for (int i = 0; i < n; i++) {
        float val = fabs(x[i]);
        if (val > wmax) wmax = val;
    }

    float scale = wmax / Q_MAX;
    qx->s = scale;

    // calculate and write the quantized values
    for (int i = 0; i < n; i++) {
        float quant_value = x[i] / scale; // scale
        int8_t quantized = (int8_t) roundf(quant_value); // round and clamp
        qx->q[i] = quantized;
    }
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    void *p = *ptr;
    QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));
    for(int i=0; i<n; i++) {
        /* read scale factor */
        res[i].s = *(float*)p;
        p = (float*)p + 1;

        /* map quantized int8 values*/
        res[i].q = (int8_t*)p;
        p = (int8_t*)p + size_each;
    }
    *ptr = p; // advance ptr to current position
    return res;
}

void memory_map_weights(TransformerWeights *w, Config* p, void* ptr, uint8_t shared_classifier) {
    int head_size = p->dim / p->n_heads;
    // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
    float* fptr = (float*) ptr; // cast our pointer to float*
    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;

    // now read all the quantized weights
    ptr = (void*)fptr; // now cast the pointer back to void*
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * head_size));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * head_size));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * head_size) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

void read_checkpoint(Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    unsigned char* curr_ptr = WEIGHTS;
    uint32_t magic_number = (uint32_t*)curr_ptr[0];
    if (magic_number != 0x616b3432) {
        printf("magic number mismatch, weights may be garbage!\n");
    }
    curr_ptr += sizeof(uint32_t);

    int64_t version = (int64_t*)curr_ptr[0];
    if (version != 3) {
        printf("Bad version, need version 3!\n");
    }
    curr_ptr += sizeof(int64_t);

    int header_size = 256; // header size of version 3
    // read in the Config
    memcpy(config, &curr_ptr, sizeof(Config));
    curr_ptr = curr_ptr + sizeof(Config);

    // read in flags
    uint8_t shared_classifier = curr_ptr; // a byte to indicate if the classifier is shared
    curr_ptr = curr_ptr + sizeof(uint8_t);

    // figure out the file size
    *file_size = WEIGHTS_LEN;

    // map the data pointer to the WEIGHTS array
    *data = (float*)WEIGHTS;
    void* weights_ptr = ((char*)*data) + header_size; // skip header bytes. char is 1 byte
    memory_map_weights(weights, config, weights_ptr, shared_classifier);
}

void build_transformer(Transformer *t) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(&t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // free QuantizedTensors
    free(t->weights.q_tokens);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if(t->weights.wcls != t->weights.q_tokens) { free(t->weights.wcls); }
    // free the RunState buffers
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// neural net blocks; the dynamics of the Transformer

void rmsnorm(float* o, float* x, float* weight, int size, bool relu) {
    // calculate sum of squares
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    if (relu) {
        for (int j = 0; j < size; j++) {
            o[j] = weight[j] * (ss * x[j]);
            o[j] = o[j] - 1.0f;
            if (o[j] < 0.0f) {
                o[j] = 0.0f;
            }
        }
    } else {
        for (int j = 0; j < size; j++) {
            o[j] = weight[j] * (ss * x[j]);
        }
    }
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// implements W is not transposed for dense matmul
void matmul_dense(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized
    
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        int j;
        for (j = 0; j <= n; j++) {
            ival += ((int32_t) x->q[j]) * ((int32_t) w->q[in + j]);
        }

        xout[i] = ((float) ival) * w->s * x->s;;
    }
}

// implements W is transposed for sparsity
void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // inputs to this function are both quantized

    // major hack... use xout float as temporary int32 accumulation space

    int32_t* xout_int = (int32_t*)xout;
    for (int i = 0; i < d; i++) xout_int[i] = 0;

    for (int i = 0; i < n; i++) {
        int offset = i * d; 
        int32_t curr_x = x->q[i];
        // if input is 0, skip
        if (curr_x!=0) {
            for (int j = 0; j <= d; j++) {
                xout_int[j] += curr_x * ((int32_t) w->q[offset + j]);
            }
        } else {
            SPARSECOUNT++;
        }
    }
    for (int i = 0; i < d; i++) xout[i] = xout_int[i] * x->s * w->s;
}

float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    //memcpy(x, w->token_embedding_table + token*dim, dim * sizeof(float));
    dequantize(w->q_tokens, x, dim, token*dim);

    // forward all the layers
    for(int l = 0; l < p->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(s->xb, x, w->rms_att_weight + l*dim, dim, true);

        // qkv matmuls for this position
        quantize(&s->xq, s->xb, dim);
        matmul(s->q, &s->xq, w->wq + l, dim, dim);
        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        
        for (int i = 0; i < kv_dim; i++) {
            int k = roundf(s->k[i] / KV_CACHE_SCALE);
            int v = roundf(s->v[i] / KV_CACHE_SCALE);
            if (k > 127) k = 127;
            if (v > 127) v = 127;
            if (k < -128) k = -128;
            if (v < -128) v = -128;
            s->key_cache[loff + pos * kv_dim + i] = k;
            s->value_cache[loff + pos * kv_dim + i] = v;
        }
        //quantize(&s->key_cache, s->k, kv_dim, loff + pos * kv_dim);
        //quantize(&s->value_cache, s->v, kv_dim, loff + pos * kv_dim);
        
        //float* key_cache_row = s->key_cache + loff + pos * kv_dim;
        //float* value_cache_row = s->value_cache + loff + pos * kv_dim;
        //memcpy(key_cache_row, s->k, kv_dim * sizeof(*key_cache_row));
        //memcpy(value_cache_row, s->v, kv_dim * sizeof(*value_cache_row));

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < p->n_heads; h++) {
            // get the query vector for this head
            float* q = s->q + h * head_size;
            // attention scores for this head
            float* att = s->att + h * p->seq_len;
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                int8_t* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                //int k_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i] * KV_CACHE_SCALE;
                    //score += q[i] * (s->key_cache.q[k_offset+i] * s->key_cache.s[(k_offset+i) / GS]);
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }

            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            // weighted sum of the values, store back into xb
            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                int8_t* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                //int v_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i] * KV_CACHE_SCALE;
                    //xb[i] += a * (s->value_cache.q[v_offset+i] * s->value_cache.s[(v_offset+i) / GS]);
                }
            }
        }

        // final matmul to get the output of the attention
        quantize(&s->xq, s->xb, dim);
        matmul(s->xb2, &s->xq, w->wo + l, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(s->xb, x, w->rms_ffn_weight + l*dim, dim, true);

        // Now for FFN in PyTorch we have: self.w2(F.relu(self.w1(x)-0.25) * self.w3(x))
        // first calculate self.w1(x), then use sparsity to calculate self.w3(x)
        quantize(&s->xq, s->xb, dim);
        matmul(s->hb, &s->xq, w->w1 + l, dim, hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, dim, hidden_dim);

        // Shifted ReLU
        for (int i = 0; i < hidden_dim; i++) {
            // changed to relu - 0.25 to encourage sparsity outlined in LLMs in a flash
            s->hb[i] -= 0.25f;
            if (s->hb[i] < 0.0f) s->hb[i] = 0.0f;
            // elementwise multiply with w3(x)
            s->hb[i] *= s->hb2[i];
        }

        // final matmul to get the output of the ffn
        quantize(&s->hq, s->hb, hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(x, x, w->rms_final_weight, dim, false);

    // classifier into logits
    quantize(&s->xq, x, dim);
    matmul_dense(s->logits, &s->xq, w->wcls, dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {
    int16_t str_size;
    int16_t vocab_size;
    char* vocab;
} Tokenizer;

void build_tokenizer(Tokenizer* t, int vocab_size) {
    unsigned char* curr_ptr = TOK;
    // read in the header
    memcpy(t, &curr_ptr, 2*sizeof(int16_t));
    if (vocab_size != t->vocab_size) { printf(stderr, "vocab size does not match\n"); exit(EXIT_FAILURE); }
    curr_ptr += 2*sizeof(int16_t);
    
    // map the rest directly in memory
    t->vocab = (char*)curr_ptr;
}

void free_tokenizer(Tokenizer* t) {
    free(t->vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab + (token * t->str_size);
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    return piece;
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    // struct timespec time;
    // clock_gettime(CLOCK_REALTIME, &time);
    // return time.tv_sec * 1000 + time.tv_nsec / 1000000;
    return CLINT->MTIME;
    // return READ_CSR("time");
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, int steps) {
    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = 1;   // kick off with the BOS(=1)
    int pos = 0;     // position in the sequence

    int kv_dim = (transformer->config.dim * transformer->config.n_kv_heads) / transformer->config.n_heads;
    int dim = transformer->config.dim;
    int head_size = dim / transformer->config.n_heads;

    while (pos < steps) {
        // shift key/value cache to the left by 1 if exceed max context
        ///*
        float* logits;
        if (pos >= transformer->config.seq_len) { 
            //printf("\n");
            // (layer, seq_len, dim)
            // kv cache is actually shuffling invariant, so we can just rotate the RoPE
            for (int l = 0; l < transformer->config.n_layers; l++) {
                int loff = l * transformer->config.seq_len * kv_dim;
                for (int j = 1; j < transformer->config.seq_len-1; j++) {
                    // shift & rotate k, simply shift v
                    // RoPE relative positional encoding: complex-valued rotate q and k in each head
                    int8_t* from = transformer->state.key_cache + loff + (j+1)*kv_dim; 
                    int8_t* to = transformer->state.value_cache + loff + j*kv_dim;
                    for (int i = 0; i < kv_dim; i+=2) {
                        int head_dim = i % head_size;
                        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
                        float val = -1 * freq; // set pos = -1 to rotate backwards by 1
                        float fcr = cosf(val);
                        float fci = sinf(val);
                        float v0 = from[i];
                        float v1 = from[i+1];
                        to[i]   = roundf((float)from[i] * fcr - (float)from[i+1] * fci);
                        to[i+1] = roundf((float)from[i] * fci + (float)from[i+1] * fcr);
                    }
                    /*
                    memcpy( transformer->state.key_cache + loff + j*kv_dim, 
                            transformer->state.key_cache + loff + (j+1)*kv_dim, 
                            kv_dim * sizeof(int8_t));
                    */
                    memcpy( transformer->state.value_cache + loff + j*kv_dim, 
                            transformer->state.value_cache + loff + (j+1)*kv_dim, 
                            kv_dim * sizeof(int8_t));
                }
            }
            // forward the transformer to get logits for the next token
            logits = forward(transformer, token, transformer->config.seq_len-1);
        } else {
            // forward the transformer to get logits for the next token
            logits = forward(transformer, token, pos);
        }
        //*/
        
        // sample the next token from the logits
        next = sample(sampler, logits);

        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        printf("%s", piece); 
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(int argc, char **argv) {
  /* USER CODE BEGIN 1 */
  uint8_t counter = 0;
  
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/
  
  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  /* USER CODE BEGIN 2 */

  // set up GPIO registers
  // GPIO_InitTypeDef GPIO_init_config;
  // GPIO_init_config.mode = GPIO_MODE_OUTPUT;
  // GPIO_init_config.pull = GPIO_PULL_NONE;
  // GPIO_init_config.drive_strength = GPIO_DS_STRONG;
  // HAL_GPIO_init(GPIOA, &GPIO_init_config, GPIO_PIN_0 | GPIO_PIN_1 | GPIO_PIN_2 | GPIO_PIN_3);

  // set up UART registers
  UART_InitTypeDef UART_init_config;
  UART_init_config.baudrate = 115200;
  UART_init_config.mode = UART_MODE_TX_RX;
  UART_init_config.stopbits = UART_STOPBITS_2;
  HAL_UART_init(UART0, &UART_init_config);

    // default parameters
  float temperature = 0.8f;              // 0.0 = greedy deterministic. 1.0 = original. don't set higher
  float topp = 0.9f;                      // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
  unsigned int steps = 512;                       // number of steps to run for
  char *prompt = "Let me tell you a story about computer.";        // prompt string
  unsigned long long rng_seed = CLINT->MTIME;        // seed rng with time by default
  char *mode = "generate";                // generate|chat
  char *system_prompt = NULL;             // the (optional) system prompt to use in chat mode

  // parameter validation/overrides

  if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
  if (temperature < 0.0) temperature = 0.0;
  if (topp < 0.0 || 1.0 < topp) topp = 0.9;
  if (steps < 0) steps = 64;

  // build the Transformer via the model .bin file
  Transformer transformer;
  build_transformer(&transformer);
  if (steps == 0) steps = transformer.config.seq_len; // ovrerride to ~max length

  // build the Tokenizer via the tokenizer .bin file
  Tokenizer tokenizer;
  build_tokenizer(&tokenizer, transformer.config.vocab_size);

  // build the Sampler
  Sampler sampler;
  build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1) {

    sampler.rng_state = CLINT->MTIME;

    // run!
    generate(&transformer, &tokenizer, &sampler, steps);

    printf("Sparse elements: %ld\n", SPARSECOUNT);
    SPARSECOUNT = 0;
    printf("========================================\n");
    HAL_delay(1000);
    /* USER CODE END WHILE */
  }
  /* USER CODE BEGIN 3 */

  /* USER CODE END 3 */
}

/*
 * Main function for secondary harts
 * 
 * Multi-threaded programs should provide their own implementation.
 */
void __attribute__((weak, noreturn)) __main(void) {
  while (1) {
   asm volatile ("wfi");
  }
}
