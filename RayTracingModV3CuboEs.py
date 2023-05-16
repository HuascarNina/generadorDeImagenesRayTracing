from PIL import Image
import numpy as np

ancho = 400
alto = 300

# posicion de la luz y color.
posicionLuz = np.array([2., 4., -5])
colorLuz = np.ones(3)

maxReflecciones = 5  # Maximo numero de luz reflectora.
col = np.zeros(3)  # Actual color.
camara = np.array([-0.1, 0.3, -1.])  # Camara.
OrigenCamara = np.array([0., 0., 0.])  # origen de la camara.
img = np.zeros((alto, ancho, 3))

# parametros de luz y material
ambiente = .05
reflexionDifusa = 1.
reflexionEspecular = 1.
especularK = 50

# coordenadas pantalla
r = float(ancho) / alto
pantalla = (-1., -1. / r + .25, 1., 1. / r + .25)


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

    vectorRayo = rayoO + t*rayoD  # punto de interseccion sobre el objeto
    vectorNormal = obtenerNormal(objeto, vectorRayo)
    color = obtenerColor(objeto, vectorRayo)
    vectorLuzE = normalizarVector(posicionLuz - vectorRayo)
    vectorOrigenE = normalizarVector(camara-vectorRayo)

    # -----------------------------------------------------------------punto de sombra
    sombra = [interseccion(vectorRayo+vectorNormal*0.0001, vectorLuzE, objetoSombra)
              for k, objetoSombra in enumerate(escena) if k != objetoX]

    if sombra and min(sombra) < np.inf:
        return

    colorRayo = ambiente
    # difuso
    colorRayo += objeto.get('reflexionDifusa', reflexionDifusa) * \
        max(np.dot(vectorNormal, vectorLuzE), 0)*color
    # especular
    colorRayo += objeto.get('reflexionEspecular', reflexionEspecular) * max(np.dot(
        vectorNormal, normalizarVector(vectorLuzE + vectorOrigenE)), 0) ** especularK * colorLuz
    return objeto, vectorRayo, vectorNormal, colorRayo


def interseccion(vectorO, vectorD, objeto):
    if objeto['type'] == 'plano':
        return interseccionPlano(vectorO, vectorD, objeto['posicion'], objeto['normal'])
    elif objeto['type'] == 'esfera':
        return interseccionEsfera(vectorO, vectorD, objeto['posicion'], objeto['radio'])


def obtenerNormal(objeto, vectorRayo):
    if objeto['type'] == 'esfera':
        vectorNormal = normalizarVector(vectorRayo-objeto['posicion'])
    elif objeto['type'] == 'plano':
        vectorNormal = objeto['normal']

    return vectorNormal


def obtenerColor(objeto, vectorNormal):
    color = objeto['color']
    if not hasattr(color, '__len__'):
        color = color(vectorNormal)
    return color


def interseccionEsfera(vectorO, vectorD, vectorS, vectorR):
    # rayo (vectorO, vecotorD)
    # esfera (vectorS,VectorR) s = punto

    # encontrado el discriminante
    a = np.dot(vectorD, vectorD)
    vectorAux = vectorO-vectorS
    b = 2*np.dot(vectorD, vectorAux)
    c = np.dot(vectorAux, vectorAux)-vectorR*vectorR

    dicriminante = b*b-4*a*c
    if dicriminante > 0:
        discriminanteRaiz = np.sqrt(dicriminante)
        q = (-b - discriminanteRaiz) / \
            2.0 if b < 0 else (-b+discriminanteRaiz)/2.0
        t0 = q/a
        t1 = c/q
        t0, t1 = min(t0, t1), max(t0, t1)
        if t1 >= 0:
            return t1 if t0 < 0 else t0
    return np.inf


def interseccionPlano(vectorO, vectorD, vectorPos, vectorNor):
    # rayo (vectorO, vecotorD)
    # plano (vectorPos, vectorNor)
    denominador = np.dot(vectorD, vectorNor)
    if np.abs(denominador) < 0:
        return np.inf
    d = np.dot(vectorPos-vectorO, vectorNor)/denominador

    if d < 0:
        return np.inf

    return d


def esfera(posicion, radio, color, reflexion):
    return dict(type='esfera', posicion=np.array(posicion), radio=np.array(radio), color=np.array(color), reflexion=reflexion)


def plano(posicion, normal):
    return dict(type='plano', posicion=np.array(posicion),
                normal=np.array(normal),
                color=lambda M: (colorPlane0
                                 if (int(M[0] * 2) % 2) == (int(M[2] * 2) % 2) else colorPlane1),
                reflexionDifusa=.75, reflexionEspecular=.5, reflexion=.25)


colorPlane0 = 1. * np.ones(3)
colorPlane1 = 0. * np.ones(3)

escena = [
    plano([0., -.5, 0.], [0., 1., 0.]),
    esfera([.25, .1, 0.01], .1, [1., 1., 0.], 0.4),
    esfera([.06, .1, 0.16], .1, [.5, .223, .5], 0.4),
    esfera([-.13, .1, 0.31], .1, [1., .572, .184], 0.4),
    esfera([-.32, .1, 0.46], .1, [.5, .223, .5], 0.4),
    esfera([-.51, .1, 0.61], .1, [1., .572, .184], 0.4),
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
        reflexion = 1.
        while reflex < maxReflecciones:
            trazoR = rayo(rayO, rayD)
            if not trazoR:
                break
            objeto, vectorRayo, vectorNormal, rayoC = trazoR
            rayO, rayD = vectorRayo + vectorNormal * .0001, normalizarVector(
                rayD - 2 * np.dot(rayD, vectorNormal) * vectorNormal)
            reflex += 1
            col += reflexion * rayoC
            reflexion *= objeto.get('reflexion', 1.)
        img[alto - j - 1, i, :] = np.clip(col, 0, 1)

print("Imagen generada exitosamente, revise su escritorio!")
guardarRayTracing = Image.fromarray((255 * img).astype(np.uint8), "RGB")
guardarRayTracing.save(
    "C:\\Users\\Huascar Nina\\Desktop\\resultadoFinCubo.png")