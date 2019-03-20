//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2018-tol.
// TILOS megvaltoztatni
//=============================================================================================
#define _USE_MATH_DEFINES		// M_PI
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

// Resolution of screen
const unsigned int windowWidth = 600, windowHeight = 600;

//--------------------------
struct vec2 {
//--------------------------
    float x, y;

    vec2(float x0 = 0, float y0 = 0) { x = x0; y = y0; }

    vec2 operator*(float a) const { return vec2(x * a, y * a); }

    vec2 operator+(const vec2& v) const { // vector + vector, color + color, point + vector
        return vec2(x + v.x, y + v.y);
    }
    vec2 operator-(const vec2& v) const { // vector - vector, color - color, point - point
        return vec2(x - v.x, y - v.y);
    }
    vec2 operator*(const vec2& v) const { return vec2(x * v.x, y * v.y); }

    vec2 operator-() const {
        return vec2(-x, -y);
    }

    void SetUniform(unsigned shaderProg, char * name) {
        int location = glGetUniformLocation(shaderProg, name);
        if (location >= 0) glUniform2fv(location, 1, &x);
        else printf("uniform %s cannot be set\n", name);
    }
};

inline float dot(const vec2& v1, const vec2& v2) {
    return (v1.x * v2.x + v1.y * v2.y);
}

inline float length(const vec2& v) { return sqrtf(dot(v, v)); }

inline vec2 normalize(const vec2& v) { return v * (1 / length(v)); }

//--------------------------
struct vec3 {
//--------------------------
    float x, y, z;

    vec3(float x0 = 0, float y0 = 0, float z0 = 0) { x = x0; y = y0; z = z0; }

    vec3(vec2 v) { x = v.x; y = v.y; z = 0; }

    vec3 operator*(float a) const { return vec3(x * a, y * a, z * a); }

    vec3 operator+(const vec3& v) const { // vector + vector, color + color, point + vector
        return vec3(x + v.x, y + v.y, z + v.z);
    }
    vec3 operator-(const vec3& v) const { // vector - vector, color - color, point - point
        return vec3(x - v.x, y - v.y, z - v.z);
    }
    vec3 operator*(const vec3& v) const { return vec3(x * v.x, y * v.y, z * v.z); }

    vec3 operator-()  const {
        return vec3(-x, -y, -z);
    }

    void SetUniform(unsigned shaderProg, char * name) {
        int location = glGetUniformLocation(shaderProg, name);
        if (location >= 0) glUniform3fv(location, 1, &x);
        else printf("uniform %s cannot be set\n", name);
    }
};

inline float dot(const vec3& v1, const vec3& v2) {
    return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

inline float length(const vec3& v) { return sqrtf(dot(v, v)); }

inline vec3 normalize(const vec3& v) { return v * (1 / length(v)); }

inline vec3 cross(const vec3& v1, const vec3& v2) {
    return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

//---------------------------
struct mat4 { // row-major matrix 4x4
//---------------------------
    float m[4][4];
public:
    mat4() {}
    mat4(float m00, float m01, float m02, float m03,
         float m10, float m11, float m12, float m13,
         float m20, float m21, float m22, float m23,
         float m30, float m31, float m32, float m33) {
        m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
        m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
        m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
        m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
    }

    mat4 operator*(const mat4& right) const {
        mat4 result;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
            }
        }
        return result;
    }

    void SetUniform(unsigned shaderProg, char * name) {
        int location = glGetUniformLocation(shaderProg, name);
        if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, &m[0][0]);
        else printf("uniform %s cannot be set\n", name);
    }
};

inline mat4 TranslateMatrix(vec3 t) {
    return mat4(1,  0,   0,   0,
                0,   1,   0,   0,
                0,   0,   1,   0,
                t.x, t.y, t.z, 1);
}

inline mat4 ScaleMatrix(vec3 s) {
    return mat4(s.x,   0,   0, 0,
                0, s.y,   0, 0,
                0,   0, s.z, 0,
                0,   0,   0, 1);
}

inline mat4 RotationMatrix(float angle, vec3 w) {
    float c = cosf(angle), s = sinf(angle);
    w = normalize(w);
    return mat4(c * (1 - w.x*w.x) + w.x*w.x, w.x*w.y*(1 - c) + w.z*s, w.x*w.z*(1 - c) - w.y*s, 0,
                w.x*w.y*(1 - c) - w.z*s, c * (1 - w.y*w.y) + w.y*w.y, w.y*w.z*(1 - c) + w.x*s, 0,
                w.x*w.z*(1 - c) + w.y*s, w.y*w.z*(1 - c) - w.x*s, c * (1 - w.z*w.z) + w.z*w.z, 0,
                0, 0, 0, 1);
}

//--------------------------
struct vec4 {
//--------------------------
    float x, y, z, w;
    vec4(float x0 = 0, float y0 = 0, float z0 = 0, float w0 = 0) {
        x = x0; y = y0; z = z0; w = w0; // vector:0, point: 1, plane: d, RGBA: opacity
    }
    vec4 operator*(float a) const { return vec4(x * a, y * a, z * a, w * a); }

    vec4 operator/(float d) const { return vec4(x / d, y / d, z / d, w / d); }

    vec4 operator+(const vec4& v) const {
        return vec4(x + v.x, y + v.y, z + v.z, w + v.w);
    }
    vec4 operator-(const vec4& v)  const {
        return vec4(x - v.x, y - v.y, z - v.z, w - v.w);
    }
    vec4 operator*(const vec4& v) const {
        return vec4(x * v.x, y * v.y, z * v.z, w * v.w);
    }

    void operator+=(const vec4 right) {
        x += right.x; y += right.y; z += right.z, w += right.z;
    }

    vec4 operator*(const mat4& mat) {
        return vec4(x * mat.m[0][0] + y * mat.m[1][0] + z * mat.m[2][0] + w * mat.m[3][0],
                    x * mat.m[0][1] + y * mat.m[1][1] + z * mat.m[2][1] + w * mat.m[3][1],
                    x * mat.m[0][2] + y * mat.m[1][2] + z * mat.m[2][2] + w * mat.m[3][2],
                    x * mat.m[0][3] + y * mat.m[1][3] + z * mat.m[2][3] + w * mat.m[3][3]);
    }

    void SetUniform(unsigned shaderProg, char * name) {
        int location = glGetUniformLocation(shaderProg, name);
        if (location >= 0) glUniform4fv(location, 1, &x);
        else printf("uniform %s cannot be set\n", name);
    }
};

inline float dot(const vec4& v1, const vec4& v2) {
    return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w);
}

//---------------------------
struct Texture {
//---------------------------
    unsigned int textureId;

    Texture() {
        glGenTextures(1, &textureId);  				// id generation
        glBindTexture(GL_TEXTURE_2D, textureId);    // binding
    }

    Texture(int width, int height, const std::vector<vec4>& image) {
        glGenTextures(1, &textureId);  				// id generation
        glBindTexture(GL_TEXTURE_2D, textureId);    // binding

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, &image[0]); // To GPU
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // sampling
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    void SetUniform(unsigned shaderProg, char * samplerName, unsigned int textureUnit = 0) {
        int location = glGetUniformLocation(shaderProg, samplerName);
        if (location >= 0) {
            glUniform1i(location, textureUnit);
            glActiveTexture(GL_TEXTURE0 + textureUnit);
            glBindTexture(GL_TEXTURE_2D, textureId);
        }
        else printf("uniform %s cannot be set\n", samplerName);
    }
};

//---------------------------
class GPUProgram {
//--------------------------
    unsigned int shaderProgramId;

    void getErrorInfo(unsigned int handle) { // shader error report
        int logLen, written;
        glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
        if (logLen > 0) {
            char * log = new char[logLen];
            glGetShaderInfoLog(handle, logLen, &written, log);
            printf("Shader log:\n%s", log);
            delete log;
        }
    }
    void checkShader(unsigned int shader, const char * message) { 	// check if shader could be compiled
        int OK;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
        if (!OK) { printf("%s!\n", message); getErrorInfo(shader); getchar(); }
    }
    void checkLinking(unsigned int program) { 	// check if shader could be linked
        int OK;
        glGetProgramiv(program, GL_LINK_STATUS, &OK);
        if (!OK) { printf("Failed to link shader program!\n"); getErrorInfo(program); getchar(); }
    }
public:
    GPUProgram() { shaderProgramId = 0; }

    unsigned int getId() { return shaderProgramId; }

    void Create(const char * const vertexSource, const char * const fragmentSource, const char * const fragmentShaderOutputName) {
        // Create vertex shader from string
        unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
        if (!vertexShader) {
            printf("Error in vertex shader creation\n");
            exit(1);
        }
        glShaderSource(vertexShader, 1, &vertexSource, NULL);
        glCompileShader(vertexShader);
        checkShader(vertexShader, "Vertex shader error");

        // Create fragment shader from string
        unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
        if (!fragmentShader) {
            printf("Error in fragment shader creation\n");
            exit(1);
        }

        glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
        glCompileShader(fragmentShader);
        checkShader(fragmentShader, "Fragment shader error");

        shaderProgramId = glCreateProgram();
        if (!shaderProgramId) {
            printf("Error in shader program creation\n");
            exit(1);
        }
        glAttachShader(shaderProgramId, vertexShader);
        glAttachShader(shaderProgramId, fragmentShader);

        // Connect the fragmentColor to the frame buffer memory
        glBindFragDataLocation(shaderProgramId, 0, fragmentShaderOutputName);	// this output goes to the frame buffer memory

        // program packaging
        glLinkProgram(shaderProgramId);
        checkLinking(shaderProgramId);

        // make this program run
        glUseProgram(shaderProgramId);
    }

    void Use() { 		// make this program run
        glUseProgram(shaderProgramId);
    }

    ~GPUProgram() { glDeleteProgram(shaderProgramId); }
};