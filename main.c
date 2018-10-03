/* #include <GL/glut.h> */
#include <GL/gl.h>
#include <GL/glu.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <fcntl.h>
#include <unistd.h>
#include <gbm.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "libbmp.h"

#define ASSERT(v,s) if (! (v)) {                \
                fprintf(stderr, s);             \
                exit(-1);                       \
        };

struct grayscale {
        char * plane;
        int width;
        int height;
};

int read_grayscale (struct grayscale * gs, const char * filename);
int free_grayscale (struct grayscale * gs) { free (gs->plane); return 0; }

int compute_cpu (const struct grayscale * gs1, const struct grayscale * gs2);
int compute_shader (const struct grayscale * gs1, const struct grayscale * gs2);

int main (int argc, char** argv) {
        struct grayscale gs1, gs2;

        if (argc != 3) {
                printf("Usage: %s [filename.bmp] [filename.bmp]\n", argv[0]);
                return -1;
        }

        if (read_grayscale (&gs1, argv[1]) != 0) {
                printf("Fail to read file 1");
                return -1;
        }

        if (read_grayscale (&gs2, argv[2]) != 0) {
                printf("Fail to read file 2");
                return -1;
        }

        compute_cpu (&gs1, &gs2);
        compute_shader(&gs1, &gs2);
}


int read_grayscale (struct grayscale * gs, const char * filename) {
        bmp_img img;
        int err = 0;
        char * blob;
        int i, j, width, height;
        bmp_pixel pixel;

        if ((err = bmp_img_read(&img, filename)) != BMP_OK) {
                printf("File reading err\n");
                goto error;
        }

        width  = img.img_header.biWidth;
        height = img.img_header.biHeight;
        printf("Size: %d; Width: %d; Height: %d\n",
               img.img_header.biSize, width, height);

        blob = (char*) malloc(width * height);

        for (i = 0; i < img.img_header.biHeight; i++) {
                for (j = 0; j < img.img_header.biWidth; j++) {
                        pixel = img.img_pixels[i][j];
                        blob[ i * width + j ] =
                                (char)(0.2126 * pixel.red + 0.7152 * pixel.green + 0.0722 * pixel.blue);
                }
        }
        gs->height = height;
        gs->width = width;
        gs->plane = blob;

        return 0;
error:
        return err;
}

#define WHT_LVL 210
#define BLK_LVL 40
#define WHT_DIFF 6
#define GRH_DIFF 2
#define KNORM 4.0
#define L_DIFF 5

#define MAX(A,B)                                \
        ({__typeof__(A) _A = (A);               \
                __typeof__(B) _B = (B);         \
                _A > _B ? _A : _B;})

#define GET_COEF(NOISE, ARRAY)                          \
        ({uint ret_val;                                 \
                if((NOISE>100) || (NOISE<0))		\
                        ret_val = 0;                    \
                else 					\
                        ret_val = ARRAY[NOISE/5];       \
                ret_val;                                \
        })

typedef struct {
        float noise;
        unsigned int right_diff;
        unsigned int down_diff;
} BLOCK;

static const uint wht_coef[20] = {6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 9, 9, 10, 12, 15, 25};
static const uint ght_coef[20] = {2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 8, 11, 21};

int compute_cpu (const struct grayscale * gs1, const struct grayscale * gs2) {
        clock_t t;
        
        uint w_blocks = gs1->width / 8;
        uint h_blocks = gs1->height / 8;
  
        long brightness = 0;
        long difference = 0;
        uint black = 0;
        uint frozen = 0;
        uint blc_counter = 0;

        uint black_bnd = 16;
        uint freez_bnd = 16;

        char* data_prev = gs2->plane;

        if (gs1->width != gs2->width
            || gs1->height != gs2->height) {
                printf("size mismatch\n");
                exit(-1);
        }

        t = clock();
        
        BLOCK * blocks = (BLOCK*) malloc (sizeof(BLOCK) * w_blocks * h_blocks);

        for (int j = 0; j < gs1->height; j++)
                for (int i = 0; i < gs1->width; i++) {
                        int ind = i + j*gs1->width;
                        uint8_t current = gs1->plane[ind];
                        uint8_t diff = 0;

                        /* eval-ting blocks inner noise */
                        if(((i+1)%8) && ((j+1)%8) &&
                           ((i+2)%8) && ((j+2)%8) &&
                           (i%8) && (j%8)) {
                                uint8_t lvl;
                                uint blc_index = (i/8) + (j/8)*w_blocks;
                                BLOCK *blc = &blocks[blc_index];
                                /* resetting block data */
                                if ((i%8 == 1) && (j%8 == 1)) {
                                        blc->noise = 0.0;
                                        blc->down_diff = 0;
                                        blc->right_diff = 0;
                                }
                                /* setting visibility lvl */
                                if ((current < WHT_LVL) && (current > BLK_LVL))
                                        lvl = GRH_DIFF;
                                else
                                        lvl = WHT_DIFF;
                                if (abs(current - gs1->plane[ind+1]) >= lvl)
                                        blc->noise += 1.0/(6.0*5.0*2.0);
                                if (abs(current - gs1->plane[ind+gs1->width]) >= lvl)
                                        blc->noise += 1.0/(6.0*5.0*2.0);
                        }
                        /* eval-ting brightness, freeze and diff */
                        brightness += current;
                        black += (current <= black_bnd) ? 1 : 0;
                        if (data_prev != NULL){
                                uint8_t current_prev = data_prev[ind];
                                diff = abs(current - current_prev);
                                difference += diff;
                                frozen += (diff <= freez_bnd) ? 1 : 0;
                                data_prev[ind] = current;
                        }
                }

        /* eval-ting borders diff */
        for (int j = 0; j < h_blocks-1; j++) {
                for (int i = 0; i < w_blocks-1; i++) {
                        int blc_index = i + j*w_blocks;
                        int ind = (i*8) + (j*8)*gs1->width;

                        uint h_noise = 100.0 * MAX(blocks[blc_index].noise, blocks[blc_index+1].noise);
                        uint v_noise = 100.0 * MAX(blocks[blc_index].noise, blocks[blc_index+w_blocks].noise);
                        uint h_wht_coef = GET_COEF(h_noise, wht_coef);
                        uint h_ght_coef = GET_COEF(h_noise, ght_coef);
                        uint v_wht_coef = GET_COEF(v_noise, wht_coef);
                        uint v_ght_coef = GET_COEF(v_noise, ght_coef);

                        for (uint orient = 0; orient <= 1; orient++) /* 0 = horiz, 1 = vert */ 
                                for (uint pix = 0; pix < 8; pix++) {
                                        uint8_t pixel, next, next_next, prev;
                                        uint coef;
                                        float denom = 0;
                                        float norm = 0;
                                        /* pixels */
                                        pixel = gs1->plane[ind + 8*(orient?gs1->width:1) + pix*(orient?1:gs1->width)];
                                        next = gs1->plane[ind + 8*(orient?gs1->width:1) + pix*(orient?1:gs1->width) - (orient?gs1->width:1)];
                                        next_next = gs1->plane[ind + 8*(orient?gs1->width:1) + pix*(orient?1:gs1->width) - (orient?(2*gs1->width):2)];
                                        prev = gs1->plane[ind + 8*(orient?gs1->width:1) + pix*(orient?1:gs1->width) + (orient?gs1->width:1)];
                                        /* coefs */
                                        if ((pixel < WHT_LVL) && (pixel > BLK_LVL))
                                                coef = orient ? v_ght_coef : h_ght_coef;
                                        else
                                                coef = orient ? v_wht_coef : h_wht_coef;
                                        /* eval */
                                        denom = roundf((float)(abs(prev - pixel) + abs(next - next_next))/KNORM);
                                        norm = (float)abs(next - pixel) / (denom == 0 ? 1 : denom);
                                        if (norm > coef) {
                                                if (orient == 0)
                                                        blocks[blc_index].right_diff += 1;
                                                else
                                                        blocks[blc_index].down_diff += 1;
                                        }
                                }
                }
        }
        /* counting visible blocks */
        for (int j = 1; j < h_blocks-1; j++) {
                for (int i = 1; i < w_blocks-1; i++) {
                        uint loc_counter = 0;
                        BLOCK* cur = &blocks[i + j*w_blocks];
                        BLOCK* upp = &blocks[i + (j-1)*w_blocks];
                        BLOCK* lef = &blocks[(i-1) + j*w_blocks];
                        if (cur->down_diff > L_DIFF)
                                loc_counter += 1;
                        if (cur->right_diff > L_DIFF)
                                loc_counter += 1;
                        if (lef->right_diff > L_DIFF)
                                loc_counter += 1;
                        if (upp->down_diff > L_DIFF)
                                loc_counter += 1;
                        if (loc_counter >= 2)
                                blc_counter += 1;
                }
        }
  
        float BLOCKY = ((float)blc_counter*100.0) / ((float)(w_blocks-2)*(float)(h_blocks-2));
        float LUMA   = (float)brightness / (gs1->height*gs1->width);
        float BLACK  = ((float)black/((float)gs1->height*(float)gs1->width))*100.0;
        float DIFF   = (float)difference / (gs1->height*gs1->width);
        float FREEZE = (frozen/(gs1->height*gs1->width))*100.0;

        t = clock() - t;
        double time_taken = ((double)t)/CLOCKS_PER_SEC;

        printf ("CPU Results: [block: %f; luma: %f; black: %f; diff: %f; freeze: %f]\n",
                BLOCKY, LUMA, BLACK, DIFF, FREEZE);
        printf("CPU took %f seconds to execute \n", time_taken);
        
        return 0;
}

static const char* shader_source =
        "#version 430\n"
        //"#extension GL_ARB_compute_shader : enable\n"
        //"#extension GL_ARB_shader_storage_buffer_object : enable\n"
        "#define WHT_LVL 210\n"
        "// 210 //0.90196078\n"
        "#define BLK_LVL 40\n"
        "// 40 //0.15625\n"
        "#define WHT_DIFF 6\n"
        "// 6 //0.0234375\n"
        "#define GRH_DIFF 2\n"
        "// 2 //0.0078125\n"
        "#define KNORM 4\n"
        "#define L_DIFF 5\n"
        "\n"
        "#define BLOCK_SIZE 8\n"
        "\n"
        "struct Noize {\n"
        "        float frozen;\n"
        "        float black;\n"
        "        float bright;\n"
        "        float diff;\n"
        "        float noize;\n"
        "        int   visible;\n"
        "};\n"
        "\n"
        "layout (r8, binding = 0) uniform image2D tex;\n"
        "layout (r8, binding = 1) uniform image2D tex_prev;\n"
        "uniform int width;\n"
        "uniform int height;\n"
        "uniform int stride;\n"
        "uniform int black_bound;\n"
        "uniform int freez_bound;\n"
        "\n"
        "layout (std430, binding=10) buffer Interm {\n"
        "         Noize noize_data [];\n"
        "};\n"
        "\n"
        "layout (local_size_x = BLOCK_SIZE, local_size_y = BLOCK_SIZE, local_size_z = 1) in; \n"
        "\n"
        "        shared int noize;"
        "        shared int bright;"
        "        shared int diff;"
        "        shared int black;"
        "        shared int frozen;"
        "\n"
        "void main() {\n"
        "        uint  block_pos = (gl_WorkGroupID.y * (width / BLOCK_SIZE)) + gl_WorkGroupID.x;\n"
        "        ivec2 pix_pos   = ivec2(gl_GlobalInvocationID.xy);\n"                      
        "\n"
        "        /* init shared*/\n"
        "        if (gl_LocalInvocationID.xy == ivec2(0,0)) {\n"
        "             noize = 0;\n"
        "             bright = 0;\n"
        "             diff = 0;\n"
        "             black = 0;\n"
        "             frozen = 0;\n"
        "        }\n"
        "        memoryBarrierShared();"
        "        barrier();"
        "\n"
        "        int   pix       = int(imageLoad(tex, pix_pos).r * 255);\n"
        "        int   pix_right = int(imageLoad(tex, ivec2(pix_pos.x + 1, pix_pos.y)).r * 255);\n"
        "        int   pix_down  = int(imageLoad(tex, ivec2(pix_pos.x, pix_pos.y + 1)).r * 255);\n"
        "        int   pix_prev  = int(imageLoad(tex_prev, pix_pos).r * 255);\n"
        "        /* Noize */\n"
        "        int   lvl = WHT_DIFF;\n"
        "        if ((pix < WHT_LVL) && (pix > BLK_LVL)) {\n"
        "              lvl = GRH_DIFF;\n"
        "        }\n"
        "        if (abs(pix - pix_right) > lvl) {\n"
        "            atomicAdd(noize, 1);\n"
        "        }\n"
        "        if (abs(pix - pix_down) > lvl) {\n"
        "            atomicAdd(noize, 1);\n"
        "        }\n"
        "        /* Brightness */\n"
        "        atomicAdd(bright, pix);\n"
        "        /* Black */\n"
        "        if (pix <= black_bound) {\n"
        "            atomicAdd(black, 1);\n"
        "        }\n"
        "        /* Diff */\n"
        "        int   diff_pix = abs(pix - pix_prev);\n"
        "        atomicAdd(diff, diff_pix);\n"
        "        /* Frozen */\n"
        //"        if (diff_pix <= freez_bound) {\n"
        "        if (pix != pix_prev) {\n"
        "            atomicAdd(frozen, 1);\n"
        "        }\n"
        "        memoryBarrierShared();"
        "        barrier();"
        "        /* Store results */\n"
        "        if (gl_LocalInvocationID.xy == ivec2(0,0)) {\n"
        "            noize_data[block_pos].noize  = float(noize) / (8.0 * 8.0 * 2.0);\n"
        "            noize_data[block_pos].black  = float(black);\n"
        "            noize_data[block_pos].frozen = float(frozen);\n"
        "            noize_data[block_pos].bright = float(bright) / 255.0;\n"
        "            noize_data[block_pos].diff   = float(diff) / 255.0;\n"
        "            noize_data[block_pos].visible = 0;\n"
        "        }\n"    
        "}\n";

static const char* shader_source2 =
        "#version 430\n"
        "#extension GL_ARB_compute_shader : enable\n"
        "#extension GL_ARB_shader_image_load_store : enable\n"
        "#extension GL_ARB_shader_storage_buffer_object : enable\n"
        "#define WHT_LVL 0.90196078\n"
        "// 210\n"
        "#define BLK_LVL 0.15625\n"
        "// 40\n"
        "#define WHT_DIFF 0.0234375\n"
        "// 6\n"
        "#define GRH_DIFF 0.0078125\n"
        "// 2\n"
        "#define KNORM 4.0\n"
        "#define L_DIFF 5\n"
        "\n"
        "#define BLOCK_SIZE 8\n"
        "\n"
        "#define BLOCKY 0\n"
        "#define LUMA 1\n"
        "#define BLACK 2\n"
        "#define DIFF 3\n"
        "#define FREEZE 4\n"
        "\n"
        "struct Noize {\n"
        "        float frozen;\n"
        "        float black;\n"
        "        float bright;\n"
        "        float diff;\n"
        "        float noize;\n"
        "        int   visible;\n"
        "};\n"
        "\n"
        "layout (r8, binding = 0) uniform image2D tex;\n"
        "\n"
        "uniform int width;\n"
        "uniform int height;\n"
        "\n"
        "layout (std430, binding=10) buffer Interm {\n"
        "         Noize noize_data [];\n"
        "};\n"
        "\n"
        "//const uint wht_coef[20] = {8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 11, 11, 12, 14, 17, 27};\n"
        "//const uint ght_coef[20] = {4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 7, 8, 10, 13, 23};\n"
        "//const uint wht_coef[20] = {10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 13, 13, 14, 16, 19, 29};	       \n"
        "//const uint ght_coef[20] = {6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 9, 9, 10, 12, 15, 25};\n"
        "const uint wht_coef[20] = {6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 9, 9, 10, 12, 15, 25};\n"
        "const uint ght_coef[20] = {2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6, 8, 11, 21};\n"
        "\n"
        "/* x = Edge pixel offset, y = Edge ID */\n"
        "layout (local_size_x = 8, local_size_y = 4, local_size_z = 1) in;\n"
        "\n"
        "float get_coef(float noize, uint array[20]) {\n"
        "        uint ret_val;                                     \n"
        "        if((noize>100) || (noize<0))\n"
        "                ret_val = 0;                              \n"
        "        else                                              \n"
        "                ret_val = array[uint(noize/5)];                 \n"
        "        return float(ret_val/255.0);\n"
        "}\n"
        "\n"
        "void main() {\n"
        "        uint block_pos = (gl_WorkGroupID.y * (width / BLOCK_SIZE)) + gl_WorkGroupID.x;\n"
        "        /* l d r u */\n"
        "        uint pixel_off = gl_LocalInvocationID.x;\n"
        "        uint edge = gl_LocalInvocationID.y;\n"
        "        /* l (-1, 0) d (0, -1) r (1, 0) u (0, 1) */ \n"
        "        ivec2 edge_off = ivec2 ( mod(edge + 1, 2) * int(edge - 1),\n"
        "                                 mod(edge, 2) * int(edge - 2));\n"
        "\n"
        "        /* Would not compute blocks near the borders */\n"
        "        if (gl_WorkGroupID.x == 0\n"
        "            || gl_WorkGroupID.x >= (width / BLOCK_SIZE)\n"
        "            || gl_WorkGroupID.y == 0\n"
        "            || gl_WorkGroupID.y >= (height / BLOCK_SIZE))\n"
        "                return;\n"
        "\n"
        "        shared int loc_counter;\n"
        "        shared int vis[4];\n"
        "        /* Noize coeffs */\n"
        "        uint pos = block_pos + edge_off.x + (edge_off.y * width / BLOCK_SIZE);\n"
        "        float noize = 100.0 * max(noize_data[block_pos].noize, noize_data[pos].noize);\n"
        "        float white = get_coef(noize, wht_coef);\n"
        "        float grey  = get_coef(noize, ght_coef);\n"
        "\n"
        "        if (gl_LocalInvocationID.x == 0) {\n"
        "\n"
        "                loc_counter = 0;\n"
        "                atomicExchange(vis[edge], 0);\n"
        "\n"
        "        }\n"
        "\n"
        "        ivec2 zero = ivec2(gl_WorkGroupID.x * BLOCK_SIZE,\n"
        "                           gl_WorkGroupID.y * BLOCK_SIZE);\n"
        "        float pixel = imageLoad(tex, ivec2(zero.x + abs(edge_off.y) * pixel_off +\n"
        "                                           (edge_off.x == 1 ? (BLOCK_SIZE-1) : 0),\n"
        "                                           zero.y + abs(edge_off.x) * pixel_off +\n"
        "                                           (edge_off.y == 1 ? (BLOCK_SIZE-1) : 0))).r;\n"
        "        float prev  = imageLoad(tex, ivec2(zero.x + abs(edge_off.y) * pixel_off + \n"
        "                                           (edge_off.x == 1 ? (BLOCK_SIZE-2) : 0) + \n"
        "                                           (edge_off.x == -1 ? 1 : 0), \n"
        "                                           zero.y + abs(edge_off.x) * pixel_off + \n"
        "                                           (edge_off.y == 1 ? (BLOCK_SIZE-2) : 0) + \n"
        "                                           (edge_off.y == 1 ? 1 : 0))).r;\n"
        "        float next  = imageLoad(tex, ivec2(zero.x + abs(edge_off.y) * pixel_off + \n"
        "                                           (edge_off.x == 1 ? BLOCK_SIZE : 0) + \n"
        "                                           (edge_off.x == -1 ? -1 : 0),           \n"
        "                                           zero.y + abs(edge_off.x) * pixel_off + \n"
        "                                           (edge_off.y == 1 ? BLOCK_SIZE : 0) + \n"
        "                                           (edge_off.y == 1 ? -1 : 0))).r;\n"
        "        float next_next  = imageLoad(tex, ivec2(zero.x + abs(edge_off.y) * pixel_off + \n"
        "                                                (edge_off.x == 1 ? (BLOCK_SIZE+1) : 0) + \n"
        "                                                (edge_off.x == -1 ? -2 : 0), \n"
        "                                                zero.y + abs(edge_off.x) * pixel_off + \n"
        "                                                (edge_off.y == 1 ? (BLOCK_SIZE+1) : 0) + \n"
        "                                                (edge_off.y == 1 ? -2 : 0))).r;\n"
        "        float coef = ((pixel < WHT_LVL) && (pixel > BLK_LVL)) ? grey : white;\n"
        "        float denom = round( (abs(prev-pixel) + abs(next-next_next)) / KNORM);\n"
        "        denom = (denom == 0.0) ? 1.0 : denom;\n"
        "        float norm = abs(next-pixel) / denom;\n"
        "        \n"
        "        if (norm > coef)\n"
        "                atomicAdd(vis[edge], 1);\n"
        "\n"
        "        /* counting visible blocks */\n"
        "\n"
        "        if (gl_LocalInvocationID.x == 0) {\n"
        "                if (vis[edge] > L_DIFF)\n"
        "                        atomicAdd(loc_counter, 1);\n"
        "        }\n"
        "\n"
        "        barrier();\n"
        "\n"
        "        if (gl_LocalInvocationID.xy == ivec2(0,0)) {\n"
        "                if (loc_counter >= 2)\n"
        "                        noize_data[block_pos].visible = 1;\n"
        "        }\n"
        "}\n";

struct Noize {
        GLfloat frozen;
        GLfloat black;
        GLfloat bright;
        GLfloat diff;
        GLfloat noize;
        GLint   visible;
};

int compute_shader (const struct grayscale * gs1, const struct grayscale * gs2) {
        clock_t t;
        int32_t fd;//, shader_fd;
        //off_t fsize;
        int result;
        //char *shader_source;//, *shader_source2;
        EGLDisplay egl_dpy;
        EGLConfig cfg;
        EGLint count;
        EGLContext ctx;

        GLuint shader, shader2;
        GLuint shader_program, shader_program2;

        if (gs1->width != gs2->width
            || gs1->height != gs2->height) {
                printf("size mismatch\n");
                exit(-1);
        }

        /* Shader 1 */
        //shader_fd = open("./shader.glsl", O_RDONLY);
        //fsize = lseek(shader_fd, 0, SEEK_END);
        //lseek(shader_fd, 0, SEEK_SET);

        //shader_source = malloc(fsize);
        //read(shader_fd, shader_source, fsize);
        //close(shader_fd);

        /* Shader 2 */
        //shader_fd = open("./shader2.glsl", O_RDONLY);
        //fsize = lseek(shader_fd, 0, SEEK_END);
        //lseek(shader_fd, 0, SEEK_SET);

        //shader_source2 = malloc(fsize);
        //read(shader_fd, shader_source2, fsize-1);
        
        fd = open("/dev/dri/renderD128", O_RDWR);

        struct gbm_device *gbm = gbm_create_device(fd);

        ASSERT(gbm,"GBM init failure\n");

        egl_dpy = eglGetPlatformDisplay(EGL_PLATFORM_GBM_MESA, gbm, NULL);

        ASSERT(egl_dpy, "Display init failure\n");

        result = eglInitialize (egl_dpy, NULL, NULL);
        ASSERT(result, "EGL init failure\n");

        static const EGLint config_attribs[] = {
                EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                EGL_NONE
        };
        result = eglChooseConfig(egl_dpy, config_attribs, &cfg, 1, &count);

        ASSERT(result, "Config failure\n");

        result = eglBindAPI(EGL_OPENGL_API);

        ASSERT(result, "Bind failure\n");
        
        static const EGLint attribs[] = {
                EGL_CONTEXT_CLIENT_VERSION, 4,
                EGL_NONE
        };
        ctx = eglCreateContext(egl_dpy, cfg, EGL_NO_CONTEXT, attribs);

        ASSERT(ctx != EGL_NO_CONTEXT, "Context init failure\n");

        result = eglMakeCurrent(egl_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx);

        ASSERT(result, "Contex make current failure\n");

        /* Shader init */
        shader = glCreateShader (GL_COMPUTE_SHADER);
        shader2 = glCreateShader (GL_COMPUTE_SHADER);
        shader_program = glCreateProgram();
        shader_program2 = glCreateProgram();

        /* Compile shader 1 */
        glShaderSource(shader, 1, (const char**)&shader_source, NULL);
        glCompileShader(shader);

        glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
        if (!result) {
                fprintf(stderr, "Error in compiling the compute shader\n");
                GLchar log[10240];
                GLsizei length;
                glGetShaderInfoLog(shader, 10239, &length, log);
                fprintf(stderr, "Compiler log:\n%s\n", log);
                exit(-1);
        }
        
        glAttachShader(shader_program, shader);
        glLinkProgram(shader_program);

        glGetProgramiv(shader_program, GL_LINK_STATUS, &result);
        if (!result) {
                fprintf(stderr, "Error in linking compute shader program\n");
                GLchar log[10240];
                GLsizei length;
                glGetProgramInfoLog(shader_program, 10239, &length, log);
                fprintf(stderr, "Linker log:\n%s\n", log);
                exit(-1);
        } 
        
        glDeleteShader(shader);

        /* Compile shader 2 */
        
        glShaderSource(shader2, 1, (const char**)&shader_source2, NULL);
        glCompileShader(shader2);

        glGetShaderiv(shader2, GL_COMPILE_STATUS, &result);
        if (!result) {
                fprintf(stderr, "Error in compiling the compute shader 2\n");
                GLchar log[10240];
                GLsizei length;
                glGetShaderInfoLog(shader2, 10239, &length, log);
                fprintf(stderr, "Compiler log:\n%s\n", log);
                exit(-1);
        }
        
        glAttachShader(shader_program2, shader2);
        glLinkProgram(shader_program2);

        glGetProgramiv(shader_program2, GL_LINK_STATUS, &result);
        if (!result) {
                fprintf(stderr, "Error in linking compute shader program 2\n");
                GLchar log[10240];
                GLsizei length;
                glGetProgramInfoLog(shader_program2, 10239, &length, log);
                fprintf(stderr, "Linker log:\n%s\n", log);
                exit(-1);
        } 
        
        glDeleteShader(shader2);

        /* Init variables */
        GLuint tex, tex_prev;
        glGenTextures(1, &tex);
        glActiveTexture(GL_TEXTURE0);
        ASSERT(glGetError() == GL_NO_ERROR, "Texture binding err\n");
        glBindTexture(GL_TEXTURE_2D, tex);
        ASSERT(glGetError() == GL_NO_ERROR, "Texture binding err\n");
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, gs1->width, gs1->height,
                     0, GL_RED, GL_UNSIGNED_BYTE, gs1->plane);
        ASSERT(glGetError() == GL_NO_ERROR, "Texture init err\n");
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);  
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);  
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glGenTextures(1, &tex_prev);
        glActiveTexture(GL_TEXTURE0 + 1);
        ASSERT(glGetError() == GL_NO_ERROR, "Texture binding err\n");
        glBindTexture(GL_TEXTURE_2D, tex_prev);
        ASSERT(glGetError() == GL_NO_ERROR, "Texture binding err\n");
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, gs2->width, gs2->height,
                     0, GL_RED, GL_UNSIGNED_BYTE, gs2->plane);
        ASSERT(glGetError() == GL_NO_ERROR, "Texture init err\n");
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);  
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);  
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        int p;
        glGetInternalformativ(GL_TEXTURE_2D, GL_RED, GL_INTERNALFORMAT_RED_SIZE, 1, &p);
        printf("RED SIZE: %x\n", p);
        glGetInternalformativ(GL_TEXTURE_2D, GL_RED, GL_INTERNALFORMAT_RED_TYPE, 1, &p);
        printf("RED FORMAT: %x, %x\n", p, GL_UNSIGNED_NORMALIZED);

        ASSERT(glGetError() == GL_NO_ERROR, "Texture binding err\n");

        GLuint noize_buffer;
        glGenBuffers(1, &noize_buffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, noize_buffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (gs1->width / 8) * (gs1->height / 8) * sizeof(struct Noize),
                     NULL, GL_DYNAMIC_COPY);
        glBindBufferBase (GL_SHADER_STORAGE_BUFFER, 10, noize_buffer);

        ASSERT(glGetError() == GL_NO_ERROR, "buf err dispatch error\n");

        t = clock();
        
        glUseProgram(shader_program);
        ASSERT(glGetError() == GL_NO_ERROR, "use err dispatch error\n");

        glBindImageTexture(0, tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);
        glBindImageTexture(1, tex_prev, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R8);

        glUniform1i(glGetUniformLocation(shader_program, "tex"), 0);
        glUniform1i(glGetUniformLocation(shader_program, "tex_prev"), 1);
        glUniform1i(glGetUniformLocation(shader_program, "width"), gs1->width);
        glUniform1i(glGetUniformLocation(shader_program, "height"), gs1->height);
        glUniform1i(glGetUniformLocation(shader_program, "stride"), gs1->width);
        glUniform1i(glGetUniformLocation(shader_program, "black_bound"), 16);
        glUniform1i(glGetUniformLocation(shader_program, "freez_bound"), 16);

        
        /* Actual computation */
        
        glDispatchCompute(gs1->width / 8, gs1->height / 8, 1);

        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        
        glUseProgram(shader_program2);

        glUniform1i(glGetUniformLocation(shader_program2, "tex"), 0);
        glUniform1i(glGetUniformLocation(shader_program2, "width"), gs1->width);
        glUniform1i(glGetUniformLocation(shader_program2, "height"), gs1->height);

        glDispatchCompute(gs1->width / 8, gs1->height / 8, 1);

        ASSERT(glGetError() == GL_NO_ERROR, "second dispatch error\n");
        
        //glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, 0);

        glMemoryBarrier(GL_ALL_BARRIER_BITS);
        glFinish();

        float blocky = 0.0, luma = 0.0, black = 0.0, diff = 0.0, freeze = 0.0;
        struct Noize * data;
        
        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, noize_buffer,
                          0, (gs1->width / 8) * (gs1->height / 8) * sizeof(struct Noize));
        data = (struct Noize *)glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                                0, (gs1->width / 8) * (gs1->height / 8) * sizeof(struct Noize),
                                                GL_MAP_READ_BIT);

        for (int i = 0; i < (gs1->width / 8) * (gs1->height / 8); i++) {
                luma   += data[i].bright;
                diff   += data[i].diff;
                freeze += data[i].frozen;
                black  += data[i].black;
                blocky += (float)data[i].visible;
        }

        luma   = 256.0 * luma / (float)(gs1->height * gs1->width);
        diff   = 100.0 * diff / (float)(gs1->height * gs1->width);
        black  = 100.0 * black / (float)(gs1->height * gs1->width);
        freeze = 100.0 * freeze / (float)(gs1->height * gs1->width);
        blocky = 100.0 * blocky / (float)(gs1->height * gs1->width / 64);
        //black  = data[(gs->width / 8) * 3 + 13].black;
        t = clock() - t;
        double time_taken = ((double)t)/CLOCKS_PER_SEC;
        
        printf ("Shader Results: [block: %f; luma: %f; black: %f; diff: %f; freeze: %f]\n",
                blocky, luma, black, diff, freeze);

        printf("GPU took %f seconds to execute \n", time_taken);

        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        // glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, 0);
        /* cleanup */
        glDeleteProgram(shader_program);
        
        eglDestroyContext(egl_dpy, ctx);
        eglTerminate(egl_dpy);
        gbm_device_destroy(gbm);
        close(fd);
        
        return 0;
}
