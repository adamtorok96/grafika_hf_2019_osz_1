//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2018. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    :
// Neptun :
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include <algorithm>
#include <complex>
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSource = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix

    layout(location = 0) in vec2 vertexPos;	// Varying input: vp = vertex position is expected in attrib array 0
    layout(location = 1) in vec3 vertexColor;

    out vec3 color;

	void main() {
		gl_Position = vec4(vertexPos.x, vertexPos.y, 0, 1) * MVP;		// transform vp from modeling space to normalized device space
        color = vertexColor;
	}
)";

const char * backgroundVertexShader = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec2 vertexUV;			// Attrib Array 1

	out vec2 texCoord;								// output attribute

	void main() {
		texCoord = vertexUV;														// copy texture coordinates
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers

	in vec3 color;
	out vec4 outColor;		// computed color of the current pixel

	void main() {
		outColor = vec4(color, 1);	// computed color is the color of the primitive
	}
)";

const char * backgroundFragmentShader = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;

	in vec2 texCoord;			// variable input: interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
        fragmentColor = texture(textureUnit, texCoord);
	}
)";


GPUProgram gpuProgram; // vertex and fragment shaders
GPUProgram backgroundProgram;

class Camera2D {
    vec2 wCenter; // center in world coordinates
    vec2 wSize;   // width and height in world coordinates

public:
    Camera2D() : wCenter(0, 0), wSize(2, 2) { }

    mat4 V() {
        return TranslateMatrix(-wCenter);
    }

    mat4 P() {
        return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y));
    }

    mat4 Vinv() {
        return TranslateMatrix(wCenter);
    }

    mat4 Pinv() {
        return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2));
    }

    void Zoom(float s) {
        wSize = wSize * s;
    }

    void Pan(vec2 t) {
        wCenter = wCenter + t;
    }
} camera;

class Object {
protected:
    const mat4 M() const {
        mat4 scaleM(
            scale.x, 0, 0, 0,
            0, scale.y, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        );

        mat4 rotateZM(
            cosf(rotate.z), sinf(rotate.z), 0, 0,
            -sinf(rotate.z), cosf(rotate.z), 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1)
        ;

        mat4 translateM(
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                position.x, position.y, 0, 1
        );

        return scaleM * rotateZM * translateM;
    }

    const mat4 Minv() const {
        return mat4(
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                position.x * (-1), position.y * (-1), 0, 1
        );
    }

    const vec2 vecTransform(const vec2 & vec) const {
        vec4 wVertex = vec4(vec.x, vec.y, 0, 1) * camera.Pinv() * camera.Vinv(); // * Minv();

        return vec2(wVertex.x, wVertex.y);
    }

public:
    vec2 scale = vec2(1, 1);
    vec2 position;
    vec3 rotate;

    virtual void Draw() const {
        mat4 MVPTransform = M() * camera.V() * camera.P();
        MVPTransform.SetUniform(gpuProgram.getId(), "MVP");
    }
};

class TexturedQuad {
    GLuint vao, vbo[2];
    vec2 vertices[4], uvs[4];

    Texture * pTexture;

    unsigned int width = 128, height = 128;

    mat4 MVP;

public:
    TexturedQuad() {
        vertices[0] = vec2(-1, -1); uvs[0] = vec2(0, 0);
        vertices[1] = vec2(1, -1);  uvs[1] = vec2(1, 0);
        vertices[2] = vec2(1, 1);   uvs[2] = vec2(1, 1);
        vertices[3] = vec2(-1, 1);  uvs[3] = vec2(0, 1);

        MVP = mat4(
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
        );
    }

    void Init() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(2, vbo);

        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(uvs), uvs, GL_STATIC_DRAW);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

        std::vector<vec4> image(width * height);

        for (unsigned int y = 0; y < height; y++) {
            for (unsigned int x = 0; x < width; x++) {
                image[y * width + x] = vec4(Mandelbrot(x, y), 0, 0, 1);
            }
        }

        pTexture = new Texture(width, height, image);
    }

    float Mandelbrot (unsigned int x, unsigned int y)  {
        std::complex<float> point((float)x / width - 1.5f, (float)y / height - 0.5f);

        std::complex<float> z(0, 0);

        unsigned int nb_iter = 0;

        while (abs (z) < 2 && nb_iter <= 34) {
            z = z * z + point;
            nb_iter++;
        }

        if (nb_iter < 34)
            return 1.0f;

        return 0.0f;
    }

    void Draw() {
        glBindVertexArray(vao);

        MVP.SetUniform(backgroundProgram.getId(), "MVP");

        pTexture->SetUniform(backgroundProgram.getId(), "textureUnit");

        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
    }
};

class VertexData {
public:

    vec2 pos;
    vec3 color;

    explicit VertexData(vec2 pos = vec2(), vec3 color = vec3()) : pos{pos}, color{color} {};
};

class KochanekBartelsCurve : public Object {
    GLuint vao, vbo;

    std::vector<vec2> controlPoints;
    std::vector<vec2> vertices;
    std::vector<VertexData> vertexData;

    const unsigned int MIN_CONTROL_POINTS = 4;

    vec2 Hermite(vec2 p0, vec2 v0, float t0, vec2 p1, vec2 v1, float t1, float t) {
        vec2 a0 = p0;
        vec2 a1 = v0;
        vec2 a2 = ((p1 - p0) * 3) - (v1 + v0 * 2);
        vec2 a3 = ((p0 - p1) * 2) + (v1 + v0);

//        printf("t: %f, t0: %f t1: %f\n", t, t0, t1);

        return a3 * pow(t - t0, 3) + a2 * pow(t - t0, 2) + a1 * (t - t0) + a0;
    }

    vec2 r(float t) {
        for(unsigned int i = 1; i < controlPoints.size() - 2; i++) {
            if( (float)i <= t && t <= (float)(i+1) ) {

                vec2 v0 = ((controlPoints[i + 1] - controlPoints[i]) + (controlPoints[i] - controlPoints[i-1])) * 0.5f;
                vec2 v1 = ((controlPoints[i + 2] - controlPoints[i + 1]) + (controlPoints[i + 1] - controlPoints[i])) * 0.5f;

                return Hermite(
                        controlPoints[i], v0, (float)i,
                        controlPoints[i + 1], v1, (float)(i + 1),
                        t
                    )
                ;
            }
        }

        return {};
    }

    void generateCurve() {
        if( controlPoints.size() < MIN_CONTROL_POINTS )
            return;

        vertices.clear();
        vertexData.clear();

        for(float t = 1.0f; t < controlPoints.size() - 2; t += 0.05f) {
            addVertex(r(t), vec3(1.0f, 0.0f, 0.0f));
        }

        loadVbo();
    }

    void addVertex(const vec2 & v, const vec3 & color) {
        vertices.emplace_back(v);
        vertexData.emplace_back(VertexData(vecTransform(v), color));
    }

    void loadVbo() {
        glBindVertexArray(vao);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        glBufferData(GL_ARRAY_BUFFER, sizeof(VertexData) * vertexData.size(), &vertexData[0], GL_DYNAMIC_DRAW);
    }

    static bool comparePos(vec2 v1, vec2 v2)
    {
        return (v1.x < v2.x);
    }

public:
    void Init() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), nullptr);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), reinterpret_cast<void*>(sizeof(vec2)));
    }

public:

    void addControlPoint(float x, float y) {
        controlPoints.emplace_back(x, y);

        std::sort(controlPoints.begin(), controlPoints.end(), comparePos);

        generateCurve();
    }

    void Draw() const override {
        if( controlPoints.size() < MIN_CONTROL_POINTS )
            return;

        Object::Draw();

        glBindVertexArray(vao);
        glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(vertexData.size()));
    }

    unsigned long getControlPointsSize() const {
        return controlPoints.size();
    }

    const std::vector<vec2> & getVertices() const {
        return vertices;
    }
};

class BicycleRoadGround : Object {
    GLuint vao, vbo;

    std::vector<VertexData> vertices;

    void generate(std::vector<vec2> const & verts) {

        vertices.clear();

        for(unsigned long i = 0; i < verts.size() - 1; i++) {
            addVertices(verts[i]);
            addVertices(vec2(verts[i].x, -1));
            addVertices(verts[i + 1]);

            addVertices(vec2(verts[i].x, -1));
            addVertices(verts[i + 1]);
            addVertices(vec2(verts[i + 1].x, -1));
        }

        loadVbo();
    }

    void addVertices(const vec2 & pos) {
        vertices.emplace_back(VertexData(vecTransform(pos), vec3(0, 1, 0)));
    }

    void loadVbo() {
        glBindVertexArray(vao);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        glBufferData(GL_ARRAY_BUFFER, sizeof(VertexData) * vertices.size(), &vertices[0], GL_DYNAMIC_DRAW);
    }
public:

    void Init() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), nullptr);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), reinterpret_cast<void*>(sizeof(vec2)));
    }

    void Draw() const override {
        if( vertices.empty() )
            return;

        Object::Draw();

        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size()));
    }

    void onControlPointAdded(unsigned long nCps, std::vector<vec2> const & verts) {
        if( nCps < 4 )
            return;

        generate(verts);
    }
};

class Cyclist : Object {
    GLuint vao[2], vbo[2];

    std::vector<VertexData> staticVertices;
    std::vector<VertexData> dynamicVertices;

    const float headRadius = 0.03f;
    const float bodyLength = 0.05f;
    const float bicycleRadius = 0.06f;

    const vec3 headColor = vec3(0, 0, 1);
    const vec3 bodyColor = vec3(0, 1, 1);
    const vec3 wheelColor = vec3(1, 0, 1);

    vec2 bicycleCenter;

    float time = 0.0f;

    void initStaticVao() {
        glBindVertexArray(vao[0]);

        glGenBuffers(1, &vbo[0]);

        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), nullptr);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), reinterpret_cast<void*>(sizeof(vec2)));

        loadStaticBuffers();
    }

    void initDynamicVao() {
        glBindVertexArray(vao[1]);

        glGenBuffers(1, &vbo[1]);

        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), nullptr);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), reinterpret_cast<void*>(sizeof(vec2)));

        loadDynamicBuffers();
    }

    void loadStaticBuffers() {
        loadHead();
        loadBody();
        loadWheel();
        loadSpoke();

        loadStaticVbo();
    }

    void loadDynamicBuffers() {
        dynamicVertices.clear();

        loadFoot();

        loadDynamicVbo();
    }

    void addStaticVertex(const vec2 & v, const vec3 & color) {
        staticVertices.emplace_back(VertexData(vecTransform(v), color));
    }

    void addDynamicVertex(const vec2 & v, const vec3 & color) {
        dynamicVertices.emplace_back(VertexData(vecTransform(v), color));
    }

    void loadHead() {
        for(unsigned int i = 0; i < 360; i++) {
            vec2 p = vec2(
                    sinf(i * M_PI / 180.0f),
                    cosf(i * M_PI / 180.0f)
            ) * headRadius;

            addStaticVertex(p, headColor);
        }
    }

    void loadBody() {
        addStaticVertex(vec2(0, headRadius * (-1)), bodyColor);
        addStaticVertex(vec2(0, (headRadius + bodyLength) * (-1)), bodyColor);
    }

    void loadWheel() {
        for(unsigned int i = 0; i < 360; i++) {
            vec2 p = vec2(
                    sinf(i * M_PI / 180.0f),
                    cosf(i * M_PI / 180.0f)
            ) * bicycleRadius;

            addStaticVertex(p + bicycleCenter, wheelColor);
        }
    }

    void loadSpoke() {
        for(unsigned int i = 0; i < 360; i += 36) {
            vec2 p = vec2(
                    sinf(i * M_PI / 180.0f + time),
                    cosf(i * M_PI / 180.0f + time)
            ) * bicycleRadius;

            addStaticVertex(vec2(), wheelColor);
            addStaticVertex(p, wheelColor);
        }
    }

    void loadFoot() {
        vec2 hipPos = vec2(0, (headRadius + bodyLength) *(-1));
        vec2 kneePos = hipPos + vec2(0.08f, sin(time) * 0.05f);

        vec2 wheelRot = vec2(
                sinf(M_PI / 180.0f + time),
                cosf(M_PI / 180.0f + time)
        ) * bicycleRadius;

        vec2 wheelPos = vec2(wheelRot + bicycleCenter);

        addDynamicVertex(hipPos, bodyColor);
        addDynamicVertex(kneePos, bodyColor);

        addDynamicVertex(kneePos, bodyColor);
        addDynamicVertex(wheelPos, bodyColor);
    }

    void loadStaticVbo() {
        glBindVertexArray(vao[0]);

        glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);

        glBufferData(GL_ARRAY_BUFFER, sizeof(VertexData) * staticVertices.size(), &staticVertices[0], GL_STATIC_DRAW);
    }

    void loadDynamicVbo() {
        glBindVertexArray(vao[1]);

        glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);

        glBufferData(GL_ARRAY_BUFFER, sizeof(VertexData) * dynamicVertices.size(), &dynamicVertices[0], GL_DYNAMIC_DRAW);
    }

    mat4 wheelM() const {
        mat4 trans = mat4(
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, bicycleCenter.y, 0, 1
        );

        mat4 rotate = mat4(
                cos(time), sin(time), 0, 0,
                sin(time) * (-1), cos(time), 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
        );

        return rotate * trans * M();
    }

public:
    Cyclist() {
        bicycleCenter = vec2(0, (headRadius + bodyLength + bicycleRadius) * (-1));
    }

    void Init() {
        glGenVertexArrays(2, &vao[0]);

        initStaticVao();
        initDynamicVao();
    }

    void Animate(float dt) {
        time = dt;

        position.x += 0.001;

        loadDynamicBuffers();
    }

    void Draw() const override {
        Object::Draw();

        glBindVertexArray(vao[0]);
        glDrawArrays(GL_LINE_LOOP, 0, 360);
        glDrawArrays(GL_LINE_STRIP, 360, 2); // it should be GL_LINES. wtf?
        glDrawArrays(GL_LINE_LOOP, 362, 360);

        mat4 MVPTransform = wheelM() * camera.V() * camera.P();
        MVPTransform.SetUniform(gpuProgram.getId(), "MVP");

        glDrawArrays(GL_LINES, 722, 360);

        Object::Draw();

        glBindVertexArray(vao[1]);

        glDrawArrays(GL_LINES, 0, 4);
    }
};

TexturedQuad background;

KochanekBartelsCurve bicycleRoad;
BicycleRoadGround bicycleRoadGround;
Cyclist cyclist;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);

    background.Init();

    bicycleRoad.Init();
    bicycleRoadGround.Init();
    cyclist.Init();

    for(float x = -1.5f; x < 1.5f; x += 0.1f) {
        bicycleRoad.addControlPoint(x, sin(x * 10) * 0.1f - 0.3f);
    }

    bicycleRoadGround.onControlPointAdded(bicycleRoad.getControlPointsSize(), bicycleRoad.getVertices());

    // create program for the GPU
    gpuProgram.Create(vertexSource, fragmentSource, "outColor");
    backgroundProgram.Create(backgroundVertexShader, backgroundFragmentShader, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0, 0, 0, 0);     // background color
    glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

    backgroundProgram.Use();
    background.Draw();

    gpuProgram.Use();

    bicycleRoad.Draw();
    bicycleRoadGround.Draw();
    cyclist.Draw();

    glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    switch(key) {
        case 's': camera.Pan(vec2(-1, 0)); break;
        case 'd': camera.Pan(vec2(+1, 0)); break;
        case 'e': camera.Pan(vec2( 0, 1)); break;
        case 'x': camera.Pan(vec2( 0,-1)); break;
        case 'z': camera.Zoom(0.9f); break;
        case 'Z': camera.Zoom(1.1f); break;

        default: break;
    }
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
    // Convert to normalized device space
    //float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
    //float cY = 1.0f - 2.0f * pY / windowHeight;

    //printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
    // Convert to normalized device space
    float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
    float cY = 1.0f - 2.0f * pY / windowHeight;

    if( button == GLUT_LEFT_BUTTON && state == GLUT_DOWN ) {
//        printf("cX: %f, cY: %f\n", cX, cY) ;
        bicycleRoad.addControlPoint(cX, cY);

        bicycleRoadGround.onControlPointAdded(bicycleRoad.getControlPointsSize(), bicycleRoad.getVertices());

        glutPostRedisplay();
    }

    /*
    char * buttonStat;
    switch (state) {
        case GLUT_DOWN: buttonStat = "pressed"; break;
        case GLUT_UP:   buttonStat = "released"; break;
    }

    switch (button) {
        case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
        case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
        case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
    }
     */
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    static float oldTime = 0.0f;

    float time = glutGet(GLUT_ELAPSED_TIME) / 1000.f; // elapsed time since the start of the program
    float dt = time - oldTime;
    oldTime = time;

    cyclist.Animate(time);

    glutPostRedisplay();
}