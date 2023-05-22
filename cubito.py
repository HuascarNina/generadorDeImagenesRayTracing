from PIL import Image
import numpy as np
import glob
import os
import platform

is_windows = platform.platform().lower().startswith("windows")

ancho = 400
alto = 300

# ancho = 1200
# alto = 720

# posicion de la luz y color.
posicionLuz = np.array([2.0, 4.0, -5])
colorLuz = np.ones(3)

maxReflecciones = 5  # Maximo numero de luz reflectora.
col = np.zeros(3)  # Actual color.
camara = np.array([-0.1, 0.3, -1.0])  # Camara.
OrigenCamara = np.array([0.0, 0.0, 0.0])  # origen de la camara.
img = np.zeros((alto, ancho, 3))

# parametros de luz y material
ambiente = 0.05
reflexionDifusa = 1.0
reflexionEspecular = 1.0
especularK = 50

# coordenadas pantalla
r = float(ancho) / alto
pantalla = (-1.0, -1.0 / r + 0.25, 1.0, 1.0 / r + 0.25)


def normalizarVector(vector):
    vector /= np.linalg.norm(vector)
    return vector


def rayo(rayoO, rayoD):
    t = np.inf  # t es el escalar
    for i, objeto in enumerate(escena):
        objetoEscena = interseccion(rayoO, rayoD, objeto)
        if objetoEscena < t:
            t, objetoX = objetoEscena, i

    if t == np.inf:
        return
    objeto = escena[objetoX]

    vectorRayo = rayoO + t * rayoD  # punto de interseccion sobre el objeto
    vectorNormal = obtenerNormal(objeto, vectorRayo)
    color = obtenerColor(objeto, vectorRayo)
    vectorLuzE = normalizarVector(posicionLuz - vectorRayo)
    vectorOrigenE = normalizarVector(camara - vectorRayo)

    # ================
    # punto de sombra
    # ================
    sombra = [
        interseccion(vectorRayo + vectorNormal * 0.0001, vectorLuzE, objetoSombra)
        for k, objetoSombra in enumerate(escena)
        if k != objetoX
    ]

    if sombra and min(sombra) < np.inf:
        return

    colorRayo = ambiente
    # difuso
    colorRayo += (
        objeto.get("reflexionDifusa", reflexionDifusa)
        * max(np.dot(vectorNormal, vectorLuzE), 0)
        * color
    )
    # especular
    colorRayo += (
        objeto.get("reflexionEspecular", reflexionEspecular)
        * max(np.dot(vectorNormal, normalizarVector(vectorLuzE + vectorOrigenE)), 0)
        ** especularK
        * colorLuz
    )
    return objeto, vectorRayo, vectorNormal, colorRayo


def interseccion(vectorO, vectorD, objeto):
    if objeto["type"] == "plano":
        return interseccionPlano(vectorO, vectorD, objeto["posicion"], objeto["normal"])
    elif objeto["type"] == "esfera":
        return interseccionEsfera(vectorO, vectorD, objeto["posicion"], objeto["radio"])
    elif objeto["type"] == "cubo":
        return interseccionCubo(vectorO, vectorD, objeto["posicion"], objeto["tamano"])
    else:
        raise ValueError("Tipo de objeto no reconocido.")


def obtenerNormal(objeto, vectorRayo):
    if objeto["type"] == "esfera":
        vectorNormal = normalizarVector(vectorRayo - objeto["posicion"])
    elif objeto["type"] == "plano":
        vectorNormal = objeto["normal"]
    elif objeto["type"] == "cubo":
        vectorNormal = obtenerNormalCubo(objeto, vectorRayo)
    else:
        raise ValueError("Tipo de objeto no reconocido.")

    return vectorNormal


def obtenerNormalCubo(objeto, vectorRayo):
    object_point = vectorRayo - objeto["posicion"]
    maxc = np.max(np.abs(object_point))
    if maxc == np.abs(object_point[0]):
        return np.array([object_point[0], 0, 0])
    elif maxc == np.abs(object_point[1]):
        return np.array([0, object_point[1], 0])
    return np.array([0, 0, object_point[2]])


def obtenerColor(objeto, vectorNormal):
    color = objeto["color"]
    if not hasattr(color, "__len__"):
        color = color(vectorNormal)
    return color


def interseccionEsfera(vectorO, vectorD, vectorS, vectorR):
    # rayo (vectorO, vecotorD)
    # esfera (vectorS,VectorR) s = punto

    # encontrado el discriminante
    a = np.dot(vectorD, vectorD)
    vectorAux = vectorO - vectorS
    b = 2 * np.dot(vectorD, vectorAux)
    c = np.dot(vectorAux, vectorAux) - vectorR * vectorR

    dicriminante = b * b - 4 * a * c
    if dicriminante > 0:
        discriminanteRaiz = np.sqrt(dicriminante)
        q = (-b - discriminanteRaiz) / 2.0 if b < 0 else (-b + discriminanteRaiz) / 2.0
        t0 = q / a
        t1 = c / q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf


def interseccionCubo(vectorO, vectorD, vectorPos, vectorTam):
    xtmin, xtmax = check_axis(vectorO[0], vectorD[0], vectorPos[0], vectorTam[0])
    ytmin, ytmax = check_axis(vectorO[1], vectorD[1], vectorPos[1], vectorTam[1])
    ztmin, ztmax = check_axis(vectorO[2], vectorD[2], vectorPos[2], vectorTam[2])
    tmin = max(xtmin, ytmin, ztmin)
    tmax = min(xtmax, ytmax, ztmax)
    if tmin > tmax:
        return np.inf
    return tmax


def check_axis(o, d, p, size):
    tmin_numerator = (p - o) - size
    tmax_numerator = (p - o) + size
    tmin, tmax = tmin_numerator / d, tmax_numerator / d
    if tmin > tmax:
        tmin, tmax = tmax, tmin
    return tmin, tmax


def interseccionPlano(vectorO, vectorD, vectorPos, vectorNor):
    # rayo (vectorO, vecotorD)
    # plano (vectorPos, vectorNor)
    denominador = np.dot(vectorD, vectorNor)
    if np.abs(denominador) < 0:
        return np.inf
    d = np.dot(vectorPos - vectorO, vectorNor) / denominador

    if d < 0:
        return np.inf

    return d


def esfera(posicion, radio, color, reflexion):
    return dict(
        type="esfera",
        posicion=np.array(posicion),
        radio=np.array(radio),
        color=np.array(color),
        reflexion=reflexion,
    )


def cubo(posicion, tamano, color, reflexion):
    return dict(
        type="cubo",
        posicion=np.array(posicion),
        tamano=np.array(tamano),
        color=np.array(color),
        reflexion=reflexion,
    )


def plano(posicion, normal):
    return dict(
        type="plano",
        posicion=np.array(posicion),
        normal=np.array(normal),
        color=lambda M: (
            colorPlane0
            if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2)
            else colorPlanePink
        ),
        reflexionDifusa=0.75,
        reflexionEspecular=0.5,
        reflexion=0.25,
    )


colorPlane0 = 1.0 * np.ones(3)
colorPlane1 = 0.0 * np.ones(3)
colorPlanePink = np.array([1.0, 0.752, 0.796])

escena = [
    plano([0.0, -0.5, 0.0], [0.0, 1.0, 0.0]),
    esfera([0.25, 0.1, 0.01], 0.1, [0.0, 1.0, 0.0], 0.6),
    esfera([0.06, 0.1, 0.16], 0.1, [1.0, 0.0, 0.0], 0.4),
    esfera([-0.13, 0.1, 0.31], 0.1, [1.0, 0.572, 0.184], 0.4),
    esfera([-0.32, 0.1, 0.46], 0.1, [1.0, 1.0, 0.0], 0.4),
    esfera([-0.51, 0.1, 0.61], 0.1, [0.5, 0.223, 0.5], 0.4),
    cubo([0.1, -0.2, 0.3], [0.2, 0.2, 0.2], [0.0, 0.0, 1.0], 0.01),
]

for i, x in enumerate(np.linspace(pantalla[0], pantalla[2], ancho)):
    if i % 10 == 0:
        print(i / float(ancho) * 100, "%")
    for j, y in enumerate(np.linspace(pantalla[1], pantalla[3], alto)):
        col[:] = 0
        OrigenCamara[:2] = (x, y)
        D = normalizarVector(OrigenCamara - camara)
        reflex = 0
        rayO, rayD = camara, D
        reflexion = 1.0
        while reflex < maxReflecciones:
            trazoR = rayo(rayO, rayD)
            if not trazoR:
                break
            objeto, vectorRayo, vectorNormal, rayoC = trazoR
            rayO, rayD = vectorRayo + vectorNormal * 0.0001, normalizarVector(
                rayD - 2 * np.dot(rayD, vectorNormal) * vectorNormal
            )
            reflex += 1
            col += reflexion * rayoC
            reflexion *= objeto.get("reflexion", 1.0)
        img[alto - j - 1, i, :] = np.clip(col, 0, 1)

# Obtener la lista de archivos en la carpeta
archivos = glob.glob(
    os.path.join("./Imagenes/", "*") if is_windows else os.path.join("./dist/", "*")
)

# Contar la cantidad de archivos
cantidad_archivos = len(archivos)
ruta = (
    f"./Imagenes/ray_tracing{cantidad_archivos+1}.png"
    if is_windows
    else f"./dist/ray_tracing{cantidad_archivos+1}.png"
)

guardarRayTracing = Image.fromarray((255 * img).astype(np.uint8), "RGB")
guardarRayTracing.save(ruta)
print("Imagen generada exitosamente, revise su escritorio!")
