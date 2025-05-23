VERTEX_SHADER = """
#version 330 core

layout (location = 0) in float aXPos;
layout (location = 1) in float aYPos;
layout (location = 2) in float aZPos;
layout (location = 3) in int aClass;
out vec4 oColor;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;
uniform vec3 uCameraPos;

vec4 classToColor(int classID) {
    if (classID == 1) return vec4(1, 1, 1, 1);         // unclassified
    else if (classID == 2) return vec4(1, 0, 0, 1);    // ground
    else if (classID == 3) return vec4(1, 1, 0, 1);    // low vegetation
    else if (classID == 4) return vec4(0, 1, 0, 1);    // medium vegetation
    else if (classID == 5) return vec4(0, 1, 1, 1);    // high vegetation
    else if (classID == 6) return vec4(0, 0, 1, 1);    // building
    else if (classID == 7) return vec4(1, 0, 1, 1);    // low point
    else return vec4(0.5, 0.5, 0.5, 1);                // other
}

void main()
{
    vec4 worldPosition = uModel * vec4(aXPos, aYPos, aZPos, 1.0);
    vec4 viewPosition = uView * worldPosition;

    float distance = length(worldPosition.xyz - uCameraPos);
    float size = 20.0 / distance;
    gl_PointSize = clamp(size, 2.0, 10.0);

    gl_Position = uProjection * uView * uModel * vec4(aXPos, aYPos, aZPos, 1.0);
    oColor = classToColor(aClass);
}
"""

FRAGMENT_SHADER = """
#version 330 core

in vec4 oColor;

void main()
{
    gl_FragColor = oColor;
}
"""
