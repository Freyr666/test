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

int compute_cpu (const struct grayscale * gs);
int compute_shader (const struct grayscale * gs);

int main (int argc, char** argv) {
        struct grayscale gs;

        if (argc != 2) {
                printf("Usage: %s [filename.bmp]\n", argv[0]);
                return -1;
        }

        if (read_grayscale (&gs, argv[1]) != 0) {
                printf("Fail to read file");
                return -1;
        }

        compute_cpu (&gs);
        compute_shader(&gs);
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

int compute_cpu (const struct grayscale * gs) {
        clock_t t;
        uint w_blocks = gs->width / 8;
        uint h_blocks = gs->height / 8;
  
        long brightness = 0;
        long difference = 0;
        uint black = 0;
        uint frozen = 0;
        uint blc_counter = 0;

        uint black_bnd = 16;
        uint freez_bnd = 16;

        char* data_prev = gs->plane;

        t = clock();
        
        BLOCK * blocks = (BLOCK*) malloc (sizeof(BLOCK) * w_blocks * h_blocks);

        for (int j = 0; j < gs->height; j++)
                for (int i = 0; i < gs->width; i++) {
                        int ind = i + j*gs->width;
                        uint8_t current = gs->plane[ind];
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
                                if (abs(current - gs->plane[ind+1]) >= lvl)
                                        blc->noise += 1.0/(6.0*5.0*2.0);
                                if (abs(current - gs->plane[ind+gs->width]) >= lvl)
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
                        int ind = (i*8) + (j*8)*gs->width;

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
                                        pixel = gs->plane[ind + 8*(orient?gs->width:1) + pix*(orient?1:gs->width)];
                                        next = gs->plane[ind + 8*(orient?gs->width:1) + pix*(orient?1:gs->width) - (orient?gs->width:1)];
                                        next_next = gs->plane[ind + 8*(orient?gs->width:1) + pix*(orient?1:gs->width) - (orient?(2*gs->width):2)];
                                        prev = gs->plane[ind + 8*(orient?gs->width:1) + pix*(orient?1:gs->width) + (orient?gs->width:1)];
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
        float LUMA   = (float)brightness / (gs->height*gs->width);
        float BLACK  = ((float)black/((float)gs->height*(float)gs->width))*100.0;
        float DIFF   = (float)difference / (gs->height*gs->width);
        float FREEZE = (frozen/(gs->height*gs->width))*100.0;

        t = clock() - t;
        double time_taken = ((double)t)/CLOCKS_PER_SEC;

        printf ("CPU Results: [block: %f; luma: %f; black: %f; diff: %f; freeze: %f]\n",
                BLOCKY, LUMA, BLACK, DIFF, FREEZE);
        printf("CPU took %f seconds to execute \n", time_taken);
        
        return 0;
}

static const char* shader_source2 =
        "#version 430\n"
        "#extension GL_ARB_compute_shader : enable\n"
        "#extension GL_ARB_shader_image_load_store : enable\n"
        "#extension GL_ARB_shader_storage_buffer_object : enable\n"
        "#ifdef GL_ES\n"
        "precision mediump float;\n"
        "#endif\n"
        "#define BLOCKY 0\n"
        "#define LUMA 1\n"
        "#define BLACK 2\n"
        "#define DIFF 3\n"
        "#define FREEZE 4\n"
        "struct Noize {\n"
        "        float frozen;\n"
        "        float black;\n"
        "        float bright;\n"
        "        float diff;\n"
        "        int   visible;\n"
        "};\n"
        "uniform int width;\n"
        "uniform int height;\n"
        "layout (std430, binding=10) buffer Interm {\n"
        "         Noize noize_data [];\n"
        "};\n"
        "layout (std430, binding=11) buffer Result {\n"
        "        float data [5];\n"
        "};\n"
        "layout (local_size_x = 1, local_size_y = 1) in;\n"
        "void main() {\n"
        "        uint block_pos = (gl_GlobalInvocationID.y * (width / 8)) + gl_GlobalInvocationID.x;\n"
        "\n"
        "        if (gl_GlobalInvocationID.xy != vec2(0,0))\n"
        "                return;\n"
        "        \n"
        "        float bright = 0.0;\n"
        "        float diff = 0.0;\n"
        "        float frozen = 0.0;\n"
        "        float black  = 0.0;\n"
        "        float blocky = 0.0;  \n"
        "        for (int i = 0; i < (height * width / 64); i++) {\n"
        "                bright += float(noize_data[i].bright);\n"
        "                diff   += noize_data[i].diff;\n"
        "                frozen += noize_data[i].frozen;\n"
        "                black  += noize_data[i].black;\n"
        "                if (noize_data[i].visible == 1)\n"
        "                        blocky += 1.0;\n"
        "        }\n"
        "        data[LUMA] = 256.0 * bright / float(height * width);\n"
        "        data[DIFF] = 100.0 * diff / float(height * width);\n"
        "        data[BLACK] = 100.0 * black / float(height * width);\n"
        "        data[FREEZE] = 100.0 * frozen / float(height * width);\n"
        "        data[0] = 100.0 * blocky / float(height * width / 64);\n"
        "}";

struct Noize {
        GLfloat frozen;
        GLfloat black;
        GLfloat bright;
        GLfloat diff;
        GLint   visible;
};

int compute_shader (const struct grayscale * gs) {
        clock_t t;
        int32_t fd, shader_fd;
        off_t fsize;
        int result;
        char *shader_source;//, *shader_source2;
        EGLDisplay egl_dpy;
        EGLConfig cfg;
        EGLint count;
        EGLContext ctx;

        GLuint shader, shader2;
        GLuint shader_program, shader_program2;

        /* Shader 1 */
        shader_fd = open("./shader.glsl", O_RDONLY);
        fsize = lseek(shader_fd, 0, SEEK_END);
        lseek(shader_fd, 0, SEEK_SET);

        shader_source = malloc(fsize);
        read(shader_fd, shader_source, fsize);
        close(shader_fd);

        /* Shader 2 */
        //shader_fd = open("./shader2.glsl", O_RDONLY);
        //fsize = lseek(shader_fd, 0, SEEK_END);
        //lseek(shader_fd, 0, SEEK_SET);

        //shader_source2 = malloc(fsize);
        //read(shader_fd, shader_source2, fsize-1);
        
        fd = open("/dev/dri/renderD129", O_RDWR);

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
        GLuint tex;
        glGenTextures(1, &tex);
        glActiveTexture(GL_TEXTURE0);
        ASSERT(glGetError() == GL_NO_ERROR, "Texture binding err\n");
        glBindTexture(GL_TEXTURE_2D, tex);
        ASSERT(glGetError() == GL_NO_ERROR, "Texture binding err\n");
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, gs->width, gs->height,
                     0, GL_RED, GL_UNSIGNED_BYTE, gs->plane);
        ASSERT(glGetError() == GL_NO_ERROR, "Texture init err\n");
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);  
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);  
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glBindImageTexture(0, tex, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R16F);

        uint p;
        glGetInternalformativ(GL_TEXTURE_2D, GL_RED, GL_INTERNALFORMAT_RED_SIZE, 1, &p);
        printf("RED SIZE: %x\n", p);
        glGetInternalformativ(GL_TEXTURE_2D, GL_RED, GL_INTERNALFORMAT_RED_TYPE, 1, &p);
        printf("RED FORMAT: %x, %x\n", p, GL_UNSIGNED_NORMALIZED);

        ASSERT(glGetError() == GL_NO_ERROR, "Texture binding err\n");

        GLuint noize_buffer;
        glGenBuffers(1, &noize_buffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, noize_buffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, (gs->width / 8) * (gs->height / 8) * sizeof(struct Noize),
                     NULL, GL_DYNAMIC_COPY);
        glBindBufferBase (GL_SHADER_STORAGE_BUFFER, 10, noize_buffer);
        
        GLfloat* data;
        GLuint buffer;
        glGenBuffers(1, &buffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        glBufferData(GL_SHADER_STORAGE_BUFFER, 5 * sizeof(GLfloat), NULL, GL_DYNAMIC_COPY);
        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 11, buffer, 0, sizeof(GLfloat) * 5);

        t = clock();
        
        glUseProgram(shader_program);

        glUniform1i(glGetUniformLocation(shader_program, "tex"), 0);
        glUniform1i(glGetUniformLocation(shader_program, "tex_prev"), 0);
        glUniform1i(glGetUniformLocation(shader_program, "width"), gs->width);
        glUniform1i(glGetUniformLocation(shader_program, "height"), gs->height);
        glUniform1i(glGetUniformLocation(shader_program, "stride"), gs->width);
        glUniform1i(glGetUniformLocation(shader_program, "black_bound"), 16);
        glUniform1i(glGetUniformLocation(shader_program, "freez_bound"), 16);

        /* Actual computation */
        
        glDispatchCompute(gs->width / 8, gs->height / 8, 1);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        
        glUseProgram(shader_program2);

        glUniform1i(glGetUniformLocation(shader_program2, "tex"), 0);
        glUniform1i(glGetUniformLocation(shader_program2, "tex_prev"), 0);
        glUniform1i(glGetUniformLocation(shader_program2, "width"), gs->width);
        glUniform1i(glGetUniformLocation(shader_program2, "height"), gs->height);
        glUniform1i(glGetUniformLocation(shader_program2, "stride"), gs->width);
        glUniform1i(glGetUniformLocation(shader_program2, "black_bound"), 16);
        glUniform1i(glGetUniformLocation(shader_program2, "freez_bound"), 16);

        glDispatchCompute(gs->width / 8, 1, 1);
        
        //glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, 0);

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        glFinish();

        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, 0, buffer, 0, sizeof(GLfloat) * 5);
        data = (float *)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeof(GLfloat) * 5, GL_MAP_READ_BIT);

        t = clock() - t;
        double time_taken = ((double)t)/CLOCKS_PER_SEC;
        
        printf ("Shader Results: [block: %f; luma: %f; black: %f; diff: %f; freeze: %f]\n",
                data[0], data[1], data[2], data[3], data[4]);

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
