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

GPUProgram gpuProgram; // vertex and fragment shaders
unsigned int vao;	   // virtual world on the GPU

void initTriangle() {
    glGenVertexArrays(1, &vao);	// get 1 vao id
    glBindVertexArray(vao);		// make it active

    unsigned int vbo;		// vertex buffer object
    glGenBuffers(1, &vbo);	// Generate 1 buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    // Geometry with 24 bytes (6 floats or 3 x 2 coordinates)
    float vertices[] = { -0.8f, -0.8f, -0.6f, 1.0f, 0.8f, -0.2f };

    printf("size of verts: %d\n", sizeof(vertices));

    glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
                 sizeof(vertices),  // # bytes
                 vertices,	      	// address
                 GL_STATIC_DRAW);	// we do not change later

    glEnableVertexAttribArray(0);  // AttribArray 0
    glVertexAttribPointer(0,       // vbo -> AttribArray 0
                          2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
                          0, NULL); 		     // stride, offset: tightly packed
}

void drawTriangle() {
    glBindVertexArray(vao);  // Draw call
    glDrawArrays(GL_TRIANGLES, 0, 3);
}

class VertexData {
public:

    vec2 pos;
    vec3 color;

    VertexData(vec2 pos = vec2(), vec3 color = vec3()) : pos{pos}, color{color} {};
};

class KochanekBartelsCurve {
    GLuint vao, vbo;

    std::vector<vec2> controlPoints;
    std::vector<VertexData> vertices;

    const unsigned int MIN_CONTROL_POINTS = 4;

    /*
    float L(unsigned int i, float t) {
        float Li = 1.0f;

        for(int j = 0; j < controlPoints.size(); j++) {
            if (j != i)
                Li *= (t - (float)j) / ((float)i - (float)j);
        }

        return Li;
    }

    vec2 r(float t) {
        vec2 v;

        for(unsigned int i = 0; i < controlPoints.size(); i++) {
            v = v + controlPoints[i] * L(i, t);
        }

//        printf("x: %f, y: %f\n", v.x, v.y);
        return v;
    }
     */

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

        for(float t = 1.0f; t < controlPoints.size() - 2; t += 0.05f) {
            vertices.emplace_back(VertexData(r(t), vec3(1.0f, 0.0f, 0.0f)));
        }

        loadVbo();
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

    void Draw() const {
        if( controlPoints.size() < MIN_CONTROL_POINTS )
            return;

        glBindVertexArray(vao);
        glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(vertices.size()));
    }

    void addControlPoint(float x, float y) {
        controlPoints.emplace_back(x, y);

        generateCurve();
    }

    unsigned long getControlPointsSize() const {
        return controlPoints.size();
    }

    const std::vector<VertexData> & getVertices() const {
        return vertices;
    }
};

class BicycleRoadGround {
    GLuint vao, vbo;

    std::vector<VertexData> vertices;

    void generate(std::vector<VertexData> const & verts) {

        vertices.clear();

        for(unsigned long i = 0; i < verts.size() - 1; i++) {
            addVertices(verts[i].pos);
            addVertices(vec2(verts[i].pos.x, -1));
            addVertices(verts[i + 1].pos);

            addVertices(vec2(verts[i].pos.x, -1));
            addVertices(verts[i + 1].pos);
            addVertices(vec2(verts[i + 1].pos.x, -1));
        }

        loadVbo();
    }

    void addVertices(const vec2 & pos) {
        vertices.emplace_back(VertexData(pos, vec3(0, 1, 0)));
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

    void Draw() {
        if( vertices.empty() )
            return;

        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size()));
    }

    void onControlPointAdded(unsigned long nCps, std::vector<VertexData> const & verts) {
        if( nCps < 4 )
            return;

        generate(verts);
    }
};

class Cyclist {
    GLuint vao[2], vbo[2];

    std::vector<VertexData> staticVertices;
    std::vector<VertexData> dynamicVertices;

    const float headRadius = 0.03f;
    const float bodyLength = 0.05f;
    const float bicycleRadius = 0.06f;

    const vec3 headColor = vec3(0, 0, 1);
    const vec3 bodyColor = vec3(0, 1, 1);
    const vec3 wheelColor = vec3(1, 0, 1);

    vec2 pos;
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

        loadStaticVbo();
    }

    void loadDynamicBuffers() {
        dynamicVertices.clear();

        loadSpoke();
        loadFoot();

        loadDynamicVbo();
    }

    void loadHead() {
        for(unsigned int i = 0; i < 360; i++) {
            vec2 p = vec2(
                    sinf(i * M_PI / 180.0f),
                    cosf(i * M_PI / 180.0f)
            ) * headRadius;

            staticVertices.emplace_back(VertexData(pos + p, headColor));
        }
    }

    void loadBody() {
        staticVertices.emplace_back(VertexData(vec2(pos.x, pos.y - headRadius), bodyColor));
        staticVertices.emplace_back(VertexData(vec2(pos.x, pos.y - headRadius - bodyLength), bodyColor));
    }

    void loadWheel() {
        for(unsigned int i = 0; i < 360; i++) {
            vec2 p = vec2(
                    sinf(i * M_PI / 180.0f),
                    cosf(i * M_PI / 180.0f)
            ) * bicycleRadius;

            staticVertices.emplace_back(VertexData(pos + p + bicycleCenter, wheelColor));
        }
    }

    void loadSpoke() {
        for(unsigned int i = 0; i < 360; i += 36) {
            vec2 p = vec2(
                    sinf(i * M_PI / 180.0f + time),
                    cosf(i * M_PI / 180.0f + time)
            ) * bicycleRadius;

            dynamicVertices.emplace_back(VertexData(pos + bicycleCenter, wheelColor));
            dynamicVertices.emplace_back(VertexData(pos + bicycleCenter + p, wheelColor));
        }
    }

    void loadFoot() {
        vec2 hipPos = vec2(pos.x, pos.y - headRadius - bodyLength);
        vec2 kneePos = hipPos + vec2(pos.x + 0.08f, sin(time) * 0.05f);

        vec2 wheelRot = vec2(
                sinf(M_PI / 180.0f + time),
                cosf(M_PI / 180.0f + time)
        ) * bicycleRadius;

        vec2 wheelPos = vec2(pos + wheelRot + bicycleCenter);

        dynamicVertices.emplace_back(VertexData(hipPos, bodyColor));
        dynamicVertices.emplace_back(VertexData(kneePos, bodyColor));

        dynamicVertices.emplace_back(VertexData(kneePos, bodyColor));
        dynamicVertices.emplace_back(VertexData(wheelPos, bodyColor));
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

public:
    Cyclist() {
        bicycleCenter = vec2(pos.x, pos.y - headRadius - bodyLength - bicycleRadius);
    }

    void Init() {
        glGenVertexArrays(2, &vao[0]);

        initStaticVao();
        initDynamicVao();
    }

    void Animate(float dt) {
        time = dt;

        loadDynamicBuffers();
    }

    void Draw() {
        glBindVertexArray(vao[0]);
        glDrawArrays(GL_LINE_LOOP, 0, 360);
        glDrawArrays(GL_LINE_STRIP, 360, 2); // it should be GL_LINES. wtf?
        glDrawArrays(GL_LINE_LOOP, 362, 360);

        glBindVertexArray(vao[1]);
        glDrawArrays(GL_LINES, 0, 360);
        glDrawArrays(GL_LINES, 360, 4);
    }
};

KochanekBartelsCurve bicycleRoad;
BicycleRoadGround bicycleRoadGround;
Cyclist cyclist;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);

    bicycleRoad.Init();
    bicycleRoadGround.Init();
    cyclist.Init();

    // create program for the GPU
    gpuProgram.Create(vertexSource, fragmentSource, "outColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0, 0, 0, 0);     // background color
    glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

    // Set color to (0, 1, 0) = green
    int location = glGetUniformLocation(gpuProgram.getId(), "color");
    glUniform3f(location, 0.0f, 1.0f, 0.0f); // 3 floats

    float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix,
                              0, 1, 0, 0,    // row-major!
                              0, 0, 1, 0,
                              0, 0, 0, 1 };

    location = glGetUniformLocation(gpuProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
    glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

    bicycleRoad.Draw();
    bicycleRoadGround.Draw();
    cyclist.Draw();

    glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
    if (key == 'd')
        glutPostRedisplay();         // if d, invalidate display, i.e. redraw
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