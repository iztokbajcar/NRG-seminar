VERTEX_SHADER = """
#version 330 core

layout (location = 0) in float aXPos;
layout (location = 1) in float aYPos;
layout (location = 2) in float aZPos;
layout (location = 3) in int aClass;
out vec4 oColor;

vec4 classToColor(int classID) {
    if (classID == 1) {  // unclassified
        return vec4(1, 1, 1, 1);
    } else if (classID == 2) {  // ground
        return vec4(1, 0, 0, 1);
    } else if (classID == 3) {  // low vegetation
        return vec4(1, 1, 0, 1);
    } else if (classID == 4) {  // medium vegetation
        return vec4(0, 1, 0, 1);
    } else if (classID == 5) {  // high vegetation
        return vec4(0, 1, 1, 1);
    } else if (classID == 6) {  // building
        return vec4(0, 0, 1, 1);
    } else if (classID == 7) {  // low point
        return vec4(1, 0, 1, 1);
    }
        
    else {
        return vec4(0.5, 0.5, 0.5, 1);
    }
}

void main()
{
    // gl_PointSize = 5.0;
    gl_Position = vec4(aXPos, aYPos, aZPos, 1.0);

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
