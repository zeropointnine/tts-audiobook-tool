import unittest

from tts_audiobook_tool.text_ops.text_normalizer import TextNormalizer, normalize_spacing_en

class TestTextNormalizer(unittest.TestCase):

    def test_normalize_common(self):
        
        items = [
            (
                "Excess  white   space  ", 
                "excess white space"
            ),
            (
                "Single quote: Here's Johnny and 'single scare quote phrase'", 
                "single quote heres johnny and single scare quote phrase"
            ),
            (
                "Weird punctuation: . ... x.x...,!a", 
                "weird punctuation x x a"
            ),
            (
                "Underscore: filenames should start with \"test_\" or end with _test", 
                "underscore filenames should start with test or end with test"
            ),
            (
                "Dashes: dashed-word emdash——emdash ... –endash––endash–", 
                "dashes dashed word emdash emdash endash endash"
            ),
            (
                "“This is too much, my love!”", 
                "this is too much my love"
            ),
            (
                "Random emojis: 😉 in the Read😉Me😉? Well... 🙂‍↔️, why not?", 
                "random emojis in the readme well why not"
            ),
            (
                "Café au lait", 
                "café au lait"
            ),
            (
                "Регулярные выражения", # ensure non-roman characters don't get stripped
                "регулярные выражения" 
            ) 
        ]
        for source, answer in items:
            result = TextNormalizer.normalize_common(source)
            print("input :",source)
            print("result:", result)
            print("answer:", answer)
            print()
            self.assertEqual(result, answer)

    def test_normalize_common_numbers_en(self):
        """
        Numbers-related transformations, utilizing TextNormalizer.normalize_numbers_en()
        """

        items = [
            ("Hello 19", "hello 19"),
            ("hello nineteen", "hello 19"),
            ("ninety", "90"),
            ("twenty one", "21"),
            ("twenty-one", "21"),

            ("99% sure", "99 sure"),
            ("ninety-nine percent sure", "99 sure"),
            ("ninety nine percent sure", "99 sure"),

            # ("I have $2005", "I have $2005"),
            # ("I have two thousand and five dollars", "I have $2005"),
            # ("I have two thousand five dollars", "I have $2005"),

            # ("There were about twenty of us: five Ambassadors, a handful of Staff, and we two.", ""),
            # ("I'd spent thousands of hours in the immer. I'd been to ports on tens of countries on tens of worlds", ""),
            # ("It was the third sixteenth of September, a Dominday.", ""),

            # ("I have two thousand five dollars", "I have $2005"),
        ]
        for inp, answer in items:
            result = TextNormalizer.normalize_common(inp, language_code="en")
            print("input :",inp)
            print("result:", result)
            print("answer:", answer)
            print()
            self.assertTrue(result == answer)

    def test_normalize_common_numbers_es(self):
        """
        Numbers-related transformations, utilizing normalize_numbers_es()
        """

        # for inp, answer in items:
        #     result = TextNormalizer.normalize_common(inp, language_code="es")
        #     print("input :",inp)
        #     print("result:", result)
        #     print("answer:", answer)
        #     print()
        #     self.assertTrue(result == answer)

        for inp in SPANISH_NUMBER_TEXT:
            result = TextNormalizer.normalize_common(inp, language_code="es")
            print("input :",inp)
            print("result:", result)
            print()


    def test_sounds_the_same(self):
        
        items = [
            ("color", "colour", True),
            ("analyze", "analyse", True),
            ("been", "bean", True),
            ("been", "bean", True),
            ("one", "Juan", False),
            ("also", "Oslo", False),
            ("apple", "orange", False),
            ("embassy town", "embassytown", True)
        ]
        for a, b, answer in items:
            result = TextNormalizer.sounds_the_same_en(a, b)
            self.assertEqual(result, answer)

    def test_normalize_spacing(self):
        items = [
            # STT breaks a compound word ("fire fly" -> "firefly")
            (
                "Look at that firefly glow.",
                "Look at that fire fly glow.",
                "Look at that firefly glow."
            ),
            # STT merges two words ("highschool" -> "high school")
            (
                "I went to high school yesterday.",
                "I went to highschool yesterday.",
                "I went to high school yesterday."
            ),
            # Non-space-related difference (should not be fixed)
            (
                "The quick brown fox.",
                "The quick fox.", 
                "The quick fox."
            )
        ]
        for source, transcript, answer in items:
          result = normalize_spacing_en(source=source, transcript=transcript)
          self.assertEqual(result, answer)

if __name__ == '__main__':
    unittest.main()

# ---

SPANISH_NUMBER_TEXT = [
    # Simple cardinal integers
    "Tengo veintidós años y mi hermano tiene diecisiete.",
    "Compró treinta y cinco naranjas en el mercado.",
    "Había ciento cuarenta y dos personas en la sala.",
    "El tren llevaba novecientos noventa y nueve pasajeros.",
    "Encontraron mil doscientos libros en el depósito.",
    "La caja contenía dos mil quinientas piezas pequeñas.",
    "Recibió nueve mil novecientos noventa y nueve votos.",

    # Numbers already written as digits
    "Tengo 22 años y mi hermano tiene 17.",
    "Compró 35 naranjas en el mercado.",
    "Había 142 personas en la sala.",
    "El tren llevaba 999 pasajeros.",
    "La caja contenía 2500 piezas pequeñas.",

    # Mixed digits and words
    "Llegaron 12 alumnos por la mañana y otros quince por la tarde.",
    "El paquete pesaba cinco kilos, pero la etiqueta decía 4 kilos.",
    "Ganó tres carreras en 2024 y dos más en dos mil veinticinco.",
    "Vivió en el piso 7 durante nueve años.",
    "El capítulo 3 empieza en la página cuarenta y dos.",

    # Years and historical references
    "Nació en mil novecientos ochenta y cuatro, en una ciudad pequeña del norte.",
    "La novela fue publicada en mil novecientos sesenta y siete.",
    "En dos mil uno, la familia se mudó a Madrid.",
    "El tratado se firmó en mil setecientos noventa y ocho.",
    "Para el año dos mil treinta, esperaban terminar la obra.",
    "El archivo llevaba la fecha 15 de marzo de 1999.",
    "El documento decía: veintidós de abril de dos mil veintiséis.",

    # Dates
    "La reunión será el cinco de mayo.",
    "Nació el veintinueve de febrero de dos mil cuatro.",
    "El contrato vence el treinta y uno de diciembre.",
    "Nos vimos el primero de enero, justo después de medianoche.",
    "El informe fue fechado el 3 de julio de 2020.",
    "Salieron el doce de octubre y regresaron el diecinueve.",

    # Times
    "Eran las ocho y media cuando empezó la tormenta.",
    "Llegó a las siete y cuarto.",
    "La función comienza a las veinte treinta.",
    "El tren sale a las 6:45 de la mañana.",
    "Volvió a casa a las dos menos cuarto.",
    "A las cero horas se cerraron las puertas.",

    # Decimals — probably risky
    "La temperatura subió tres coma cinco grados.",
    "El tanque contenía doce punto siete litros.",
    "La distancia era de cero coma ocho kilómetros.",
    "El resultado fue 3,5 según el primer medidor.",
    "La máquina marcó 12.7 litros, aunque nadie le creyó.",
    "El índice bajó de dos coma uno a uno coma nueve.",

    # Money
    "Pagó veinte euros por el libro usado.",
    "La entrada costaba doce con cincuenta.",
    "Me debe cien pesos desde la semana pasada.",
    "El recibo decía 14,99 euros.",
    "La deuda ascendía a mil quinientos dólares.",
    "Solo tenía cinco céntimos en el bolsillo.",

    # Measurements
    "El muro medía tres metros de alto.",
    "La cuerda tenía veinticinco centímetros de largo.",
    "El niño pesaba dieciocho kilos.",
    "Añadió dos litros de agua y medio kilo de harina.",
    "La carretera se extendía durante ciento veinte kilómetros.",
    "La caja pesaba 2,4 kilos.",

    # Fractions and ambiguous “medio”
    "Tomó medio vaso de agua antes de salir.",
    "Necesitamos una taza y media de leche.",
    "Quedaba un cuarto de hora para el inicio.",
    "Comió tres cuartos de la tortilla.",
    "La mitad de los asistentes se fue temprano.",
    "Dividieron la herencia en dos partes iguales.",

    # Ordinals — risky / likely skip initially
    "Vivía en el tercer piso de un edificio antiguo.",
    "Fue la primera vez que vio el mar.",
    "El segundo intento salió mejor que el primero.",
    "Llegó en décimo lugar.",
    "El cuarto capítulo es el más extraño.",
    "Carlos V reinó sobre un imperio enorme.",
    "Felipe II aparece mencionado en el manuscrito.",

    # Articles that should probably not become numbers
    "Un hombre cruzó la plaza sin mirar atrás.",
    "Una mujer esperaba junto a la puerta.",
    "Un día volveré a este lugar.",
    "Una vez, cuando era niño, escuché esa misma canción.",
    "Un poco más tarde, todos guardaron silencio.",
    "Una de las cartas estaba escrita con tinta azul.",

    # “Uno/una” where normalization may or may not be okay
    "Uno de ellos levantó la mano.",
    "Solo quedaba uno en la caja.",
    "Una cayó al suelo y las otras siguieron rodando.",
    "El número uno estaba escrito en rojo.",
    "Contó uno, dos, tres, y abrió la puerta.",

    # Ranges
    "La caminata duró entre dos y tres horas.",
    "Necesitamos de quince a veinte voluntarios.",
    "El precio oscilaba entre 30 y 40 euros.",
    "Había entre cien y ciento veinte personas.",
    "Los niños tenían entre cinco y ocho años.",
    "El capítulo cubre las páginas cuarenta y dos a cincuenta.",

    # Repetition and counting dialogue
    "—Cuenta hasta diez —dijo ella. —Uno, dos, tres, cuatro...",
    "Repitió tres veces la misma frase: no, no y no.",
    "El niño murmuraba: seis, siete, ocho, nueve.",
    "Primero golpeó una vez; luego, dos veces más.",
    "Cinco, cuatro, tres, dos, uno: la sala quedó a oscuras.",

    # Large numbers
    "La ciudad tenía un millón de habitantes.",
    "El presupuesto superaba los dos millones de euros.",
    "Se vendieron ciento cincuenta mil ejemplares.",
    "La distancia era de trescientos mil kilómetros.",
    "El informe mencionaba 1.200.000 personas afectadas.",
    "La empresa perdió mil doscientos millones en un solo año.",

    # Numbers with punctuation nearby
    "Tenía veintidós, quizá veintitrés años.",
    "Compró cuatro: dos para ella y dos para su hermano.",
    "El marcador terminó tres a cero.",
    "La nota decía: “Trae seis panes, ocho huevos y dos botellas”.",
    "¿Dijiste treinta o trece?",
    "No eran cincuenta; eran quince.",

    # Potential Whisper confusions
    "Dijo treinta, pero todos entendieron trece.",
    "Pidió sesenta sillas, no setenta.",
    "La clave era dieciséis, no diecisiete.",
    "Anotó doscientos, aunque quizá dijo dos cientos.",
    "El anciano pronunció veintidós de una forma casi inaudible.",
    "Susurró mil nueve, pero la grabación parecía decir mil nueve mil.",

    # Good torture-test paragraph
    (
        "El veintidós de abril de dos mil veintiséis, a las ocho y media "
        "de la mañana, llegaron treinta y cinco cajas al almacén. Cada caja "
        "pesaba dos coma cinco kilos, aunque el recibo decía 2,4. En total, "
        "el encargado contó ochocientas setenta y cinco piezas, pero solo "
        "registró 870 porque cinco estaban rotas. “Un error así no importa”, "
        "dijo uno de los ayudantes, pero la jefa respondió que un error "
        "pequeño puede costar mil euros."
    ),

    # Conservative expected-normalization subset
    "veintidós",
    "diecisiete",
    "treinta y cinco",
    "ciento cuarenta y dos",
    "novecientos noventa y nueve",
    "mil doscientos",
    "dos mil quinientas",
    "nueve mil novecientos noventa y nueve",
    "dos mil veintiséis",
    "ochocientas setenta y cinco",

    # Initially do not normalize, or normalize only after real Whisper evidence
    "un",
    "una",
    "uno de ellos",
    "una vez",
    "medio vaso",
    "una taza y media",
    "un cuarto de hora",
    "tercer piso",
    "primera vez",
    "Carlos V",
    "doce con cincuenta",
    "tres coma cinco",
    "doce punto siete",
]        
