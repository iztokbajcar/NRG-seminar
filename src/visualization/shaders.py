VERTEX_SHADER = """
#version 120

attribute float aXPos;
attribute float aYPos;
attribute float aZPos;
attribute float aClass;
attribute float aLOD;
varying vec4 oColor;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform vec3 uCameraPos;
uniform bool uDrawLOD;

vec4 classToColor(float classID) {
    float classID_mod8 = mod(classID, 8.0);

    if (classID_mod8 == 1.0) return vec4(1, 1, 1, 1);         // unclassified
    else if (classID_mod8 == 2.0) return vec4(1, 0, 0, 1);    // ground
    else if (classID_mod8 == 3.0) return vec4(1, 1, 0, 1);    // low vegetation
    else if (classID_mod8 == 4.0) return vec4(0, 1, 0, 1);    // medium vegetation
    else if (classID_mod8 == 5.0) return vec4(0, 1, 1, 1);    // high vegetation
    else if (classID_mod8 == 6.0) return vec4(0, 0, 1, 1);    // building
    else if (classID_mod8 == 7.0) return vec4(1, 0, 1, 1);    // low point
    else return vec4(0.5, 0.5, 0.5, 1);     // other
}

void main()
{
    vec4 worldPosition = uModel * vec4(aXPos, aYPos, aZPos, 1.0);
    float distance = length(worldPosition.xyz - uCameraPos);
    float size = 20.0 / distance;
    gl_PointSize = clamp(size, 2.0, 10.0);

    gl_Position = uProjection * uView * uModel * vec4(aXPos, aYPos, aZPos, 1.0);

    if (uDrawLOD) {
        oColor = classToColor(aLOD);
    } else {
        oColor = classToColor(aClass);
    }
}
"""

FRAGMENT_SHADER = """
#version 120

varying vec4 oColor;

void main()
{
    gl_FragColor = oColor;
}
"""
