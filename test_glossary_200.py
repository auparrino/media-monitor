"""
Stress-test the glossary (Actores Clave) with 200 simulated headlines.
Validates that:
  - Non-person entities (countries, institutions) are excluded
  - Roster roles/bios/countries flow through correctly
  - International entries are capped
  - Cono Sur countries have adequate representation
  - No "Public figure" for roster-matched people
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from datetime import datetime, timedelta
from collections import Counter

from categorizer import analyze_article
from dashboard import (
    cluster_stories,
    _llm_extract_glossary,
    _scan_titles_for_known_people,
    _cluster_country_assignment,
    _normalize_person_name,
)
from name_utils import looks_like_person_name

# ═══════════════════════════════════════════════════════════════════════
# 200 SIMULATED HEADLINES
# ═══════════════════════════════════════════════════════════════════════

ARTICLES_200 = [
    # ── ARGENTINA: Milei + economia (12 arts, 4 sources) ──
    {"title": "Milei viaja a Washington para negociar nuevo acuerdo con el FMI", "source": "Infobae", "country": "argentina"},
    {"title": "El presidente Milei se reunira con autoridades del Fondo Monetario Internacional", "source": "La Nacion", "country": "argentina"},
    {"title": "Argentina busca cerrar acuerdo con el FMI por USD 20.000 millones", "source": "Clarin", "country": "argentina"},
    {"title": "Negociaciones entre Argentina y el FMI entran en etapa final segun Caputo", "source": "Ambito", "country": "argentina"},
    {"title": "Caputo confirma avance en negociaciones con el FMI en Washington", "source": "Infobae", "country": "argentina"},
    {"title": "Milei y Caputo buscan destrabar el desembolso del FMI antes de mayo", "source": "La Nacion", "country": "argentina"},
    {"title": "Milei anuncio que Argentina saldra del cepo cambiario en los proximos meses", "source": "Clarin", "country": "argentina"},
    {"title": "El gobierno de Milei confirmo la eliminacion gradual del cepo al dolar", "source": "Ambito", "country": "argentina"},
    {"title": "Milei presento nuevo plan economico con reformas estructurales para 2026", "source": "Infobae", "country": "argentina"},
    {"title": "El equipo economico de Milei negocia con bonistas internacionales en Nueva York", "source": "La Nacion", "country": "argentina"},
    {"title": "Caputo viajo a Nueva York para reunirse con fondos de inversion", "source": "Pagina 12", "country": "argentina"},
    {"title": "El ministro Caputo aseguro que la inflacion bajara al 1% mensual en julio", "source": "El Destape", "country": "argentina"},

    # ── ARGENTINA: Karina Milei + politica interna (6 arts) ──
    {"title": "Karina Milei se reunio con gobernadores del interior para negociar presupuesto", "source": "Infobae", "country": "argentina"},
    {"title": "La secretaria general Karina Milei encabezo reunion con gobernadores peronistas", "source": "La Nacion", "country": "argentina"},
    {"title": "Karina Milei amplia su poder en el armado politico del oficialismo", "source": "Clarin", "country": "argentina"},
    {"title": "La hermana del presidente consolida su rol como operadora politica clave", "source": "Pagina 12", "country": "argentina"},
    {"title": "Karina Milei y los gobernadores: la negociacion por la coparticipacion", "source": "El Destape", "country": "argentina"},
    {"title": "Karina Milei cerro acuerdo con tres gobernadores por la ley de presupuesto", "source": "Ambito", "country": "argentina"},

    # ── ARGENTINA: Macri (5 arts) ──
    {"title": "Macri critico al gobierno de Milei por el manejo de la deuda externa", "source": "Infobae", "country": "argentina"},
    {"title": "Mauricio Macri rompio el silencio y cuestiono la politica fiscal del gobierno", "source": "La Nacion", "country": "argentina"},
    {"title": "Macri se reunio con dirigentes del PRO para definir estrategia electoral", "source": "Clarin", "country": "argentina"},
    {"title": "El expresidente Macri pidio mas dialogo entre el gobierno y la oposicion", "source": "Pagina 12", "country": "argentina"},
    {"title": "Macri y Bullrich se reunieron para analizar el futuro del PRO", "source": "El Destape", "country": "argentina"},

    # ── ARGENTINA: Kicillof (5 arts) ──
    {"title": "Kicillof anuncio plan de obras publicas para la provincia de Buenos Aires", "source": "Infobae", "country": "argentina"},
    {"title": "El gobernador Kicillof presento programa de empleo para jovenes bonaerenses", "source": "La Nacion", "country": "argentina"},
    {"title": "Kicillof critico recortes del gobierno nacional a las transferencias provinciales", "source": "Pagina 12", "country": "argentina"},
    {"title": "Axel Kicillof inauguro hospital en La Matanza con fondos provinciales", "source": "El Destape", "country": "argentina"},
    {"title": "Kicillof pidio a Nacion que restituya fondos de coparticipacion a Buenos Aires", "source": "Ambito", "country": "argentina"},

    # ── ARGENTINA: Bullrich seguridad (4 arts) ──
    {"title": "Bullrich anuncio nuevos operativos de seguridad en el conurbano bonaerense", "source": "Infobae", "country": "argentina"},
    {"title": "La ministra Bullrich presento plan contra el narcotrafico en Rosario", "source": "La Nacion", "country": "argentina"},
    {"title": "Bullrich desplego fuerzas federales en la frontera norte de Argentina", "source": "Clarin", "country": "argentina"},
    {"title": "Patricia Bullrich confirmo detencion de narcos en operativo conjunto", "source": "Ambito", "country": "argentina"},

    # ── ARGENTINA: Wado de Pedro oposicion (3 arts) ──
    {"title": "Wado de Pedro propuso crear frente opositor amplio para las legislativas", "source": "Pagina 12", "country": "argentina"},
    {"title": "Eduardo de Pedro busca liderar la oposicion peronista unificada", "source": "El Destape", "country": "argentina"},
    {"title": "De Pedro se reunio con sindicalistas y movimientos sociales en la CGT", "source": "Infobae", "country": "argentina"},

    # ── ARGENTINA: Villarruel (3 arts) ──
    {"title": "Victoria Villarruel preside sesion del Senado sobre reforma previsional", "source": "La Nacion", "country": "argentina"},
    {"title": "Villarruel convoco a sesion especial del Senado para debatir ley de seguridad", "source": "Clarin", "country": "argentina"},
    {"title": "La vicepresidenta Villarruel recibio a empresarios en el Senado", "source": "Infobae", "country": "argentina"},

    # ── ARGENTINA: economia/inflacion (6 arts) ──
    {"title": "Inflacion de marzo fue del 3,2% segun el INDEC y acumula 52% interanual", "source": "Clarin", "country": "argentina"},
    {"title": "El INDEC informo que la inflacion de marzo se ubico en 3,2 por ciento", "source": "La Nacion", "country": "argentina"},
    {"title": "Inflacion: el dato de marzo mostro desaceleracion respecto a febrero", "source": "Ambito", "country": "argentina"},
    {"title": "El riesgo pais bajo a 800 puntos y el mercado celebra senales del FMI", "source": "Infobae", "country": "argentina"},
    {"title": "Bonos argentinos suben tras anuncio de acuerdo con el FMI", "source": "La Nacion", "country": "argentina"},
    {"title": "El dolar oficial se mantuvo estable en la primera semana del nuevo esquema", "source": "Ambito", "country": "argentina"},

    # ── URUGUAY: Orsi (8 arts) ──
    {"title": "Orsi presento su plan de gobierno para los primeros 100 dias", "source": "El Pais UY", "country": "uruguay"},
    {"title": "El presidente Orsi anuncio reformas en el gabinete ministerial", "source": "El Observador", "country": "uruguay"},
    {"title": "Orsi impulsa cambios en el ministerio de economia de Uruguay", "source": "La Diaria", "country": "uruguay"},
    {"title": "Orsi se reunio con el canciller para definir agenda de politica exterior", "source": "El Pais UY", "country": "uruguay"},
    {"title": "El presidente Orsi firmo decreto de emergencia habitacional", "source": "El Observador", "country": "uruguay"},
    {"title": "Orsi viajara a Brasil para reunirse con Lula la proxima semana", "source": "Montevideo Portal", "country": "uruguay"},
    {"title": "Orsi anuncio plan de viviendas sociales para 50.000 familias uruguayas", "source": "La Diaria", "country": "uruguay"},
    {"title": "El gobierno de Orsi busca fortalecer relaciones con China y Estados Unidos", "source": "El Pais UY", "country": "uruguay"},

    # ── URUGUAY: Oddone economia (4 arts) ──
    {"title": "Oddone presento el plan fiscal del nuevo gobierno ante inversores", "source": "El Observador", "country": "uruguay"},
    {"title": "El ministro Oddone anuncio medidas para contener el deficit fiscal", "source": "El Pais UY", "country": "uruguay"},
    {"title": "Oddone confirmo que Uruguay mantendra su grado inversor", "source": "La Diaria", "country": "uruguay"},
    {"title": "Gabriel Oddone se reunio con el directorio del Banco Central", "source": "Montevideo Portal", "country": "uruguay"},

    # ── URUGUAY: Cosse (3 arts) ──
    {"title": "Carolina Cosse asumio como nueva ministra de industria de Uruguay", "source": "El Pais UY", "country": "uruguay"},
    {"title": "Cosse presento plan industrial para reactivar las fabricas uruguayas", "source": "El Observador", "country": "uruguay"},
    {"title": "La ministra Cosse anuncio acuerdos con empresas tecnologicas internacionales", "source": "La Diaria", "country": "uruguay"},

    # ── URUGUAY: Lacalle Pou oposicion (3 arts) ──
    {"title": "Lacalle Pou critico la gestion economica del nuevo gobierno de Orsi", "source": "El Pais UY", "country": "uruguay"},
    {"title": "El expresidente Lacalle Pou pidio mas transparencia en el gasto publico", "source": "El Observador", "country": "uruguay"},
    {"title": "Lacalle Pou y la coalicion multicolor definen estrategia opositora", "source": "Montevideo Portal", "country": "uruguay"},

    # ── URUGUAY: legislativo (3 arts) ──
    {"title": "Senado uruguayo aprobo ley de reforma del sistema de jubilaciones", "source": "El Pais UY", "country": "uruguay"},
    {"title": "Diputados uruguayos debaten proyecto de ley de educacion publica", "source": "La Diaria", "country": "uruguay"},
    {"title": "El parlamento uruguayo aprobo en general la ley de medios de comunicacion", "source": "El Observador", "country": "uruguay"},

    # ── PARAGUAY: Pena (8 arts) ──
    {"title": "Pena firmo decreto para reducir aranceles a la importacion en Paraguay", "source": "ABC Color", "country": "paraguay"},
    {"title": "El presidente Pena anuncia plan de reduccion arancelaria para Paraguay", "source": "Ultima Hora", "country": "paraguay"},
    {"title": "Paraguay: gobierno de Pena busca atraer inversion con rebaja de aranceles", "source": "5Dias", "country": "paraguay"},
    {"title": "Pena se reunio con empresarios brasileros para promover inversiones", "source": "La Nacion PY", "country": "paraguay"},
    {"title": "Santiago Pena inauguro nueva planta industrial en Ciudad del Este", "source": "ABC Color", "country": "paraguay"},
    {"title": "Pena firmo acuerdo de cooperacion energetica con Brasil", "source": "Ultima Hora", "country": "paraguay"},
    {"title": "El presidente Pena presento plan de modernizacion de la administracion publica", "source": "Hoy", "country": "paraguay"},
    {"title": "Pena anuncio creacion de zona franca en la frontera con Argentina", "source": "La Nacion PY", "country": "paraguay"},

    # ── PARAGUAY: Alliana (4 arts) ──
    {"title": "Pedro Alliana fue reelecto como presidente del Partido Colorado", "source": "ABC Color", "country": "paraguay"},
    {"title": "Alliana consolida su liderazgo en la ANR tras la convencion partidaria", "source": "Ultima Hora", "country": "paraguay"},
    {"title": "El titular de la ANR Pedro Alliana nego divisiones internas en el partido", "source": "La Nacion PY", "country": "paraguay"},
    {"title": "Alliana se reunio con Pena para definir agenda legislativa del oficialismo", "source": "Hoy", "country": "paraguay"},

    # ── PARAGUAY: Velazquez (3 arts) ──
    {"title": "Velazquez impulsa ley de transparencia en la Camara de Senadores", "source": "ABC Color", "country": "paraguay"},
    {"title": "El senador Velazquez presento proyecto de reforma del sistema judicial", "source": "La Nacion PY", "country": "paraguay"},
    {"title": "Velazquez critico la falta de control en las empresas publicas paraguayas", "source": "Ultima Hora", "country": "paraguay"},

    # ── PARAGUAY: economia/Itaipu (3 arts) ──
    {"title": "Paraguay renegocia tarifa de Itaipu con Brasil en busca de mejores condiciones", "source": "ABC Color", "country": "paraguay"},
    {"title": "Negociaciones por Itaipu avanzan con propuesta paraguaya de tarifa indexada", "source": "La Nacion PY", "country": "paraguay"},
    {"title": "El gobierno paraguayo busca duplicar los ingresos por energia de Itaipu", "source": "Hoy", "country": "paraguay"},

    # ── INTERNACIONAL: Trump (10 arts from multiple countries) ──
    {"title": "Trump anuncio nuevos aranceles del 25% contra productos de China", "source": "Infobae", "country": "argentina"},
    {"title": "Estados Unidos impone aranceles del 25% a importaciones chinas por orden de Trump", "source": "El Observador", "country": "uruguay"},
    {"title": "Guerra comercial: Trump sube aranceles a China al 25 por ciento", "source": "ABC Color", "country": "paraguay"},
    {"title": "Trump amenazo con nuevas sanciones a paises que comercien con Iran", "source": "La Nacion", "country": "argentina"},
    {"title": "El presidente Trump firmo orden ejecutiva contra inmigrantes indocumentados", "source": "Clarin", "country": "argentina"},
    {"title": "Trump critico a la Organizacion Mundial del Comercio en discurso en la Casa Blanca", "source": "Infobae", "country": "argentina"},
    {"title": "Trump y Xi Jinping acordaron nueva ronda de negociaciones comerciales", "source": "El Pais UY", "country": "uruguay"},
    {"title": "Trump busca alianza con India para contrarrestar influencia china en Asia", "source": "La Nacion PY", "country": "paraguay"},
    {"title": "Donald Trump anuncio retiro de tropas de bases militares en Europa", "source": "La Nacion", "country": "argentina"},
    {"title": "Trump impulsa recortes de impuestos para la clase media estadounidense", "source": "Ambito", "country": "argentina"},

    # ── INTERNACIONAL: Orban (5 arts) ──
    {"title": "Viktor Orban enfrenta protestas masivas en Budapest contra su gobierno", "source": "Infobae", "country": "argentina"},
    {"title": "Orban pierde apoyo en las encuestas ante avance de la oposicion hungara", "source": "La Nacion", "country": "argentina"},
    {"title": "Miles de hungaros marcharon contra el gobierno de Orban en Budapest", "source": "El Observador", "country": "uruguay"},
    {"title": "Peter Magyar se perfila como principal rival de Orban en las elecciones", "source": "ABC Color", "country": "paraguay"},
    {"title": "La oposicion hungara liderada por Magyar supera a Orban en las encuestas", "source": "La Nacion PY", "country": "paraguay"},

    # ── INTERNACIONAL: Fujimori Peru (5 arts) ──
    {"title": "Keiko Fujimori lidera encuestas para las elecciones presidenciales de Peru", "source": "Infobae", "country": "argentina"},
    {"title": "Fujimori promete mano dura contra la inseguridad si llega a la presidencia", "source": "La Nacion", "country": "argentina"},
    {"title": "Elecciones en Peru: Fujimori aventaja a sus rivales en las encuestas", "source": "El Observador", "country": "uruguay"},
    {"title": "Peru vota hoy: Keiko Fujimori es la favorita segun los sondeos", "source": "ABC Color", "country": "paraguay"},
    {"title": "Keiko Fujimori voto en las elecciones presidenciales peruanas", "source": "La Nacion PY", "country": "paraguay"},

    # ── INTERNACIONAL: Papa Leon XIV (4 arts) ──
    {"title": "El papa Leon XIV pidio paz en Medio Oriente en su primera misa solemne", "source": "Infobae", "country": "argentina"},
    {"title": "Leon XIV recibio a lideres mundiales en el Vaticano en su primera semana", "source": "La Nacion", "country": "argentina"},
    {"title": "El nuevo papa Leon XIV convoco a dialogo interreligioso global", "source": "El Observador", "country": "uruguay"},
    {"title": "Leon XIV anuncio reformas en la estructura financiera del Vaticano", "source": "ABC Color", "country": "paraguay"},

    # ── INTERNACIONAL: Ucrania (4 arts) ──
    {"title": "Rusia lanzo nuevo ataque con misiles contra infraestructura energetica de Ucrania", "source": "Infobae", "country": "argentina"},
    {"title": "Ucrania reporta ataque masivo ruso contra la red electrica del pais", "source": "La Nacion", "country": "argentina"},
    {"title": "Zelensky pide mas ayuda militar a la OTAN tras ataque ruso", "source": "El Observador", "country": "uruguay"},
    {"title": "Putin advirtio consecuencias si la OTAN sigue expandiendose hacia el este", "source": "ABC Color", "country": "paraguay"},

    # ── INTERNACIONAL: Netanyahu (3 arts) ──
    {"title": "Netanyahu nombro nuevo jefe del estado mayor del ejercito israeli", "source": "Infobae", "country": "argentina"},
    {"title": "Benjamin Netanyahu enfrenta nuevas protestas en Tel Aviv por la reforma judicial", "source": "La Nacion", "country": "argentina"},
    {"title": "Netanyahu se reunio con el secretario de Estado de Estados Unidos", "source": "El Observador", "country": "uruguay"},

    # ── REGIONAL: Mercosur (4 arts from 3 countries) ──
    {"title": "Mercosur y Union Europea avanzan en acuerdo comercial historico", "source": "Infobae", "country": "argentina"},
    {"title": "Uruguay celebra avance del acuerdo Mercosur-UE como oportunidad para exportadores", "source": "El Pais UY", "country": "uruguay"},
    {"title": "Paraguay espera beneficios del acuerdo Mercosur-UE para el sector agricola", "source": "ABC Color", "country": "paraguay"},
    {"title": "Cancilleres del Mercosur se reunieron en Montevideo para definir postura comun", "source": "La Diaria", "country": "uruguay"},

    # ── REGIONAL: energia/Itaipu-Yacireta (3 arts) ──
    {"title": "Argentina y Paraguay firmaron nuevo acuerdo sobre Yacireta", "source": "Infobae", "country": "argentina"},
    {"title": "Pena y Milei acordaron revision de tarifa de Yacireta en cumbre bilateral", "source": "ABC Color", "country": "paraguay"},
    {"title": "La energia de Yacireta sera renegociada entre Argentina y Paraguay", "source": "La Nacion PY", "country": "paraguay"},

    # ── NOISE: should be filtered ──
    {"title": "Dolar hoy: a cuanto cotiza este viernes 11 de abril", "source": "Infobae", "country": "argentina"},
    {"title": "Horoscopo de hoy viernes 11 de abril de 2025", "source": "Clarin", "country": "argentina"},
    {"title": "Quiniela Nacional: resultados del sorteo vespertino", "source": "Infobae", "country": "argentina"},
    {"title": "Boca vs River en vivo: segui el superclasico minuto a minuto", "source": "Clarin", "country": "argentina"},
    {"title": "Clima en Buenos Aires hoy: pronostico para este sabado 12 de abril", "source": "La Nacion", "country": "argentina"},
    {"title": "Dolar blue hoy sabado 12 de abril: a cuanto cotiza", "source": "Ambito", "country": "argentina"},
    {"title": "Resultado de la loteria nacional del sabado 12 de abril", "source": "Infobae", "country": "argentina"},
    {"title": "Futbol en vivo: todos los partidos de hoy en Argentina", "source": "Clarin", "country": "argentina"},

    # ── FILLER: miscellaneous domestic stories (single-source, pad to 200) ──
    {"title": "Jubilados protestaron frente al Congreso por aumento insuficiente", "source": "Pagina 12", "country": "argentina"},
    {"title": "Argentina exporto record de carne vacuna en el primer trimestre", "source": "Infobae", "country": "argentina"},
    {"title": "El Banco Central mantuvo la tasa de interes de referencia sin cambios", "source": "Ambito", "country": "argentina"},
    {"title": "Docentes bonaerenses anunciaron paro de 48 horas por salarios", "source": "La Nacion", "country": "argentina"},
    {"title": "YPF anuncio descubrimiento de nuevo pozo petrolero en Vaca Muerta", "source": "Clarin", "country": "argentina"},
    {"title": "La cosecha de soja en Argentina sera la segunda mejor de la historia", "source": "Infobae", "country": "argentina"},
    {"title": "El gobierno argentino lanzo programa de creditos para PyMEs industriales", "source": "El Destape", "country": "argentina"},
    {"title": "Aerolineas Argentinas incorporo nuevas rutas a Europa y Asia", "source": "La Nacion", "country": "argentina"},

    {"title": "Trabajadores del transporte uruguayo anunciaron paro general", "source": "El Pais UY", "country": "uruguay"},
    {"title": "Uruguay registro record de exportaciones de carne en el primer trimestre", "source": "El Observador", "country": "uruguay"},
    {"title": "La sequia afecta a productores rurales en el norte de Uruguay", "source": "La Diaria", "country": "uruguay"},
    {"title": "Montevideo sera sede del proximo foro de innovacion tecnologica regional", "source": "Montevideo Portal", "country": "uruguay"},
    {"title": "Uruguay lanza programa de becas para estudiantes de medicina", "source": "El Pais UY", "country": "uruguay"},
    {"title": "UTE anuncio inversion en parques eolicos para duplicar capacidad renovable", "source": "El Observador", "country": "uruguay"},

    {"title": "Campesinos paraguayos marcharon por reforma agraria en Asuncion", "source": "ABC Color", "country": "paraguay"},
    {"title": "Paraguay registro crecimiento del PIB del 4,5% en el primer trimestre", "source": "La Nacion PY", "country": "paraguay"},
    {"title": "El gobierno paraguayo lanzo plan contra la deforestacion en el Chaco", "source": "Hoy", "country": "paraguay"},
    {"title": "La soja paraguaya alcanzo precio record en el mercado internacional", "source": "5Dias", "country": "paraguay"},
    {"title": "ANDE invierte en modernizacion de la red electrica del pais", "source": "Ultima Hora", "country": "paraguay"},
    {"title": "Banco Nacional de Fomento otorgo creditos a pequenos productores", "source": "La Nacion PY", "country": "paraguay"},

    # ── More international filler ──
    {"title": "Lula viajo a Beijing para reunirse con Xi Jinping sobre comercio bilateral", "source": "Infobae", "country": "argentina"},
    {"title": "Pedro Sanchez visita China para estrechar relaciones comerciales", "source": "El Observador", "country": "uruguay"},
    {"title": "Elecciones en Colombia: refuerzan seguridad de candidatos presidenciales", "source": "ABC Color", "country": "paraguay"},
    {"title": "El FMI proyecto crecimiento del 2,8% para America Latina en 2026", "source": "La Nacion", "country": "argentina"},
    {"title": "La OPEP decidio mantener los recortes de produccion de petroleo", "source": "Ambito", "country": "argentina"},
    {"title": "Elon Musk presento nueva version del cohete Starship en Texas", "source": "Infobae", "country": "argentina"},
    {"title": "Macron y Scholz se reunieron en Paris para hablar sobre defensa europea", "source": "La Nacion", "country": "argentina"},
    {"title": "Maria Corina Machado denuncio persecucion politica en Venezuela", "source": "Infobae", "country": "argentina"},
    {"title": "Maduro anuncio nuevo plan economico para Venezuela ante crisis", "source": "El Observador", "country": "uruguay"},
    {"title": "Bolivia enfrenta crisis de divisas y escasez de combustibles", "source": "La Nacion PY", "country": "paraguay"},
    {"title": "Chile aprobo reforma tributaria con apoyo transversal en el Congreso", "source": "Clarin", "country": "argentina"},
    {"title": "Costa Rica se posiciona como hub tecnologico de Centroamerica", "source": "La Nacion", "country": "argentina"},

    # ── Tricky non-person entities that should NOT appear as people ──
    {"title": "Estados Unidos endurece politica migratoria en la frontera sur", "source": "Infobae", "country": "argentina"},
    {"title": "Reino Unido firmo acuerdo comercial post-Brexit con Australia", "source": "La Nacion", "country": "argentina"},
    {"title": "Naciones Unidas pidio alto el fuego inmediato en Gaza", "source": "Clarin", "country": "argentina"},
    {"title": "La Casa Blanca emitio comunicado sobre las negociaciones comerciales", "source": "El Observador", "country": "uruguay"},
    {"title": "Union Europea aprobo paquete de sanciones contra Rusia", "source": "ABC Color", "country": "paraguay"},
    {"title": "La Santa Sede convoco a conferencia interreligiosa en Roma", "source": "La Nacion", "country": "argentina"},
    {"title": "Buenos Aires sera sede de la cumbre de lideres del G20 en noviembre", "source": "Infobae", "country": "argentina"},
    {"title": "Casa Rosada confirmo la visita del presidente de Francia a Buenos Aires", "source": "La Nacion", "country": "argentina"},

    # ── Extra filler to reach 200 ──
    {"title": "Massa critico la politica monetaria del gobierno de Milei", "source": "Pagina 12", "country": "argentina"},
    {"title": "Sergio Massa pidio debate publico sobre el acuerdo con el FMI", "source": "El Destape", "country": "argentina"},
    {"title": "Grabois marcho contra los recortes sociales del gobierno nacional", "source": "Pagina 12", "country": "argentina"},
    {"title": "CFK publico carta abierta criticando la politica economica del gobierno", "source": "El Destape", "country": "argentina"},
    {"title": "Cristina Fernandez de Kirchner convoco a la unidad del peronismo", "source": "Pagina 12", "country": "argentina"},
    {"title": "Moyano amenazo con paro general si no se reabren las paritarias", "source": "Infobae", "country": "argentina"},
    {"title": "Hugo Moyano se reunio con otros lideres sindicales en la CGT", "source": "Clarin", "country": "argentina"},
    {"title": "Sturzenegger presento nuevo paquete de desregulacion economica", "source": "La Nacion", "country": "argentina"},
    {"title": "El ministro Sturzenegger anuncio eliminacion de tramites burocraticos", "source": "Ambito", "country": "argentina"},
    {"title": "Adorni confirmo en conferencia de prensa las medidas del gobierno", "source": "Infobae", "country": "argentina"},
    {"title": "Manuel Adorni desestimo criticas de la oposicion al plan economico", "source": "La Nacion", "country": "argentina"},
    {"title": "Adorni anuncio nuevas medidas de austeridad en el gasto publico", "source": "Clarin", "country": "argentina"},
    {"title": "Lacalle Pou viajo a Europa para promover inversiones en Uruguay", "source": "El Pais UY", "country": "uruguay"},
    {"title": "Orsi recibio a empresarios chinos interesados en invertir en Uruguay", "source": "El Observador", "country": "uruguay"},
    {"title": "La oposicion uruguaya cuestiono el presupuesto quinquenal de Orsi", "source": "La Diaria", "country": "uruguay"},
    {"title": "UPM confirmo ampliacion de su planta de celulosa en Uruguay", "source": "El Pais UY", "country": "uruguay"},
    {"title": "Pena inauguro ruta que conecta Asuncion con Ciudad del Este", "source": "ABC Color", "country": "paraguay"},
    {"title": "El gobierno paraguayo nego aumento de impuestos para el proximo ano", "source": "Ultima Hora", "country": "paraguay"},
    {"title": "Diputados paraguayos aprobaron reforma de la ley de empleo publico", "source": "La Nacion PY", "country": "paraguay"},
    {"title": "Sindicatos paraguayos reclamaron aumento del salario minimo para 2026", "source": "Hoy", "country": "paraguay"},
    {"title": "Milei recibio a inversores japoneses en la Casa Rosada", "source": "Infobae", "country": "argentina"},
    {"title": "El gobierno argentino lanzo licitacion para obra publica en el NOA", "source": "La Nacion", "country": "argentina"},
    {"title": "Francos coordino reunion de gabinete sobre plan de infraestructura", "source": "Clarin", "country": "argentina"},
    {"title": "El jefe de gabinete Francos se reunio con gobernadores del norte", "source": "Ambito", "country": "argentina"},
    {"title": "Orsi anuncio acuerdo con BID para financiar infraestructura vial", "source": "Montevideo Portal", "country": "uruguay"},
    {"title": "Pena y Lula firmaron acuerdo de cooperacion en salud publica", "source": "5Dias", "country": "paraguay"},
    {"title": "Trump amenazo con aranceles a Mexico si no frena la migracion", "source": "Infobae", "country": "argentina"},
    {"title": "La cumbre del G7 abordo la crisis climatica y la seguridad alimentaria", "source": "La Nacion", "country": "argentina"},
    {"title": "India supero a China como el pais mas poblado del mundo segun la ONU", "source": "El Observador", "country": "uruguay"},
    {"title": "Japon anuncio plan de estimulo economico por 200.000 millones de dolares", "source": "ABC Color", "country": "paraguay"},
    {"title": "Emmanuel Macron propuso crear fondo europeo de defensa comun", "source": "La Nacion", "country": "argentina"},
]

assert len(ARTICLES_200) == 200, f"Expected 200 articles, got {len(ARTICLES_200)}"

# ═══════════════════════════════════════════════════════════════════════
# BUILD DATAFRAME
# ═══════════════════════════════════════════════════════════════════════

def build_test_df():
    now = datetime.now()
    records = []
    for i, art in enumerate(ARTICLES_200):
        records.append({
            "id": i + 1,
            "source": art["source"],
            "country": art["country"],
            "title": art["title"],
            "url": f"https://example.com/{i}",
            "published_at": now - timedelta(hours=i * 0.3),
            "scraped_at": now - timedelta(minutes=i * 5),
            "category": "",
            "subcategory": None,
            "category_confidence": 0.0,
            "fetch_method": "rss",
            "section_url": "",
        })
    df = pd.DataFrame(records)
    for idx, row in df.iterrows():
        analysis = analyze_article(row["title"], row["url"], source=row["source"])
        df.at[idx, "category"] = analysis["category"] if analysis["accepted"] else "otros"
        df.at[idx, "subcategory"] = analysis.get("subcategory")
        df.at[idx, "category_confidence"] = analysis.get("confidence", 0.0)
    return df


# ═══════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "#" * 75)
    print("#  GLOSSARY STRESS TEST — 200 simulated headlines")
    print("#" * 75)

    # ── 1. Non-person phrase filter ──
    print("\n" + "=" * 75)
    print("TEST 1: NON-PERSON PHRASE FILTER")
    print("=" * 75)
    non_persons = [
        "Estados Unidos", "Reino Unido", "Naciones Unidas", "Casa Blanca",
        "Union Europea", "Santa Sede", "Buenos Aires", "Casa Rosada",
        "Ciudad del Este", "Costa Rica",
    ]
    persons = [
        "Javier Milei", "Donald Trump", "Keiko Fujimori", "Santiago Pena",
        "Victoria Villarruel", "Axel Kicillof", "Pedro Alliana",
    ]
    t1_ok = True
    for name in non_persons:
        result = looks_like_person_name(name)
        mark = "OK" if not result else "FAIL"
        if result:
            t1_ok = False
        print(f"  [{mark}] {name:25s} -> {result}  (expect False)")
    for name in persons:
        result = looks_like_person_name(name)
        mark = "OK" if result else "FAIL"
        if not result:
            t1_ok = False
        print(f"  [{mark}] {name:25s} -> {result}  (expect True)")
    print(f"\n  Result: {'PASS' if t1_ok else 'FAIL'}")

    # ── 2. Build clusters ──
    print("\n" + "=" * 75)
    print("TEST 2: CLUSTERING 200 ARTICLES")
    print("=" * 75)
    df = build_test_df()
    print(f"  Articles: {len(df)}")
    clusters = cluster_stories(df)
    multi = [c for c in clusters if c["multi"] and not c.get("noise")]
    single = [c for c in clusters if not c["multi"]]
    noise = [c for c in clusters if c.get("noise")]
    print(f"  Clusters: {len(clusters)} total | {len(multi)} multi | {len(single)} single | {len(noise)} noise")

    # ── 3. Entity detection scan ──
    print("\n" + "=" * 75)
    print("TEST 3: ENTITY DETECTION (scan_titles_for_known_people)")
    print("=" * 75)
    all_titles = [a["title"] for a in ARTICLES_200]
    people = _scan_titles_for_known_people(all_titles)
    print(f"\n  Detected {len(people)} people:\n")
    print(f"  {'Name':30s} | {'Country':14s} | {'Mentions':>8s} | {'Conf':>5s} | {'Basis'}")
    print(f"  {'-'*30}-+-{'-'*14}-+-{'-'*8}-+-{'-'*5}-+-{'-'*15}")
    for p in people[:25]:
        print(f"  {p['name']:30s} | {p.get('country','?'):14s} | {p['mentions']:8d} | {p.get('confidence',0):5.2f} | {p.get('match_basis','?')}")

    # ── 4. Glossary extraction (without LLM — seed-only path) ──
    print("\n" + "=" * 75)
    print("TEST 4: GLOSSARY EXTRACTION (seed path, no LLM clusters)")
    print("=" * 75)
    glossary = _llm_extract_glossary(all_titles, clusters=None)
    print(f"\n  Glossary entries: {len(glossary)}\n")
    print(f"  {'Name':30s} | {'Role':35s} | {'Country':14s} | {'Mentions':>8s}")
    print(f"  {'-'*30}-+-{'-'*35}-+-{'-'*14}-+-{'-'*8}")
    t4_issues = []
    for e in glossary:
        role = (e.get("role") or "?")[:35]
        print(f"  {e['name']:30s} | {role:35s} | {e.get('country','?'):14s} | {e.get('mentions',0):8d}")
        if e.get("role", "").lower() == "public figure":
            t4_issues.append(f"  [WARN] '{e['name']}' has generic 'Public figure' role")
        if e.get("country", "") == "regional" and e["name"] not in ("FMI",):
            t4_issues.append(f"  [WARN] '{e['name']}' country='regional' (should be resolved)")

    # ── 5. Glossary with clusters (normal path) ──
    print("\n" + "=" * 75)
    print("TEST 5: GLOSSARY EXTRACTION (with clusters, normal path)")
    print("=" * 75)
    glossary_titles = []
    seen = set()
    for cl in clusters:
        if cl.get("noise"):
            continue
        for art in cl["articles"]:
            t = art.title.strip()
            if t and t not in seen:
                seen.add(t)
                glossary_titles.append(t)
    glossary_cl = _llm_extract_glossary(glossary_titles, clusters=clusters)
    print(f"\n  Glossary entries: {len(glossary_cl)}\n")
    print(f"  {'Name':30s} | {'Role':35s} | {'Country':14s} | {'Mentions':>8s} | {'Basis'}")
    print(f"  {'-'*30}-+-{'-'*35}-+-{'-'*14}-+-{'-'*8}-+-{'-'*15}")
    for e in glossary_cl:
        role = (e.get("role") or "?")[:35]
        basis = e.get("match_basis", "?")
        print(f"  {e['name']:30s} | {role:35s} | {e.get('country','?'):14s} | {e.get('mentions',0):8d} | {basis}")

    # ── 6. Validation checks ──
    print("\n" + "=" * 75)
    print("TEST 6: VALIDATION CHECKS")
    print("=" * 75)

    t6_ok = True

    # 6a: No non-person entities
    non_person_names = {"estados unidos", "reino unido", "naciones unidas", "casa blanca",
                        "union europea", "santa sede", "buenos aires", "casa rosada"}
    for g in [glossary, glossary_cl]:
        for e in g:
            norm = _normalize_person_name(e["name"])
            if norm in non_person_names:
                print(f"  [FAIL] Non-person entity in glossary: {e['name']}")
                t6_ok = False

    if t6_ok:
        print("  [OK] No non-person entities in glossary")

    # 6b: International cap (max 3)
    for label, g in [("seed-only", glossary), ("with-clusters", glossary_cl)]:
        intl = [e for e in g if e.get("country") == "international"]
        if len(intl) <= 3:
            print(f"  [OK] International cap respected in {label}: {len(intl)} entries")
        else:
            print(f"  [FAIL] International cap exceeded in {label}: {len(intl)} entries (max 3)")
            t6_ok = False

    # 6c: Cono Sur representation (>=2 per country)
    for label, g in [("seed-only", glossary), ("with-clusters", glossary_cl)]:
        cc = Counter(e.get("country", "") for e in g)
        for country in ("argentina", "uruguay", "paraguay"):
            count = cc.get(country, 0)
            if count >= 2:
                print(f"  [OK] {country} has {count} entries in {label}")
            else:
                print(f"  [FAIL] {country} has only {count} entries in {label} (need >=2)")
                t6_ok = False

    # 6d: No "Public figure" for roster-matched people
    for label, g in [("seed-only", glossary), ("with-clusters", glossary_cl)]:
        public_figs = [e for e in g if (e.get("role") or "").lower() == "public figure"]
        if not public_figs:
            print(f"  [OK] No generic 'Public figure' roles in {label}")
        else:
            for pf in public_figs:
                print(f"  [WARN] '{pf['name']}' has 'Public figure' role in {label}")

    print(f"\n  Result: {'PASS' if t6_ok else 'SOME ISSUES'}")

    # ── Summary ──
    print("\n" + "=" * 75)
    print("SUMMARY")
    print("=" * 75)
    print(f"  T1 Non-person filter:    {'PASS' if t1_ok else 'FAIL'}")
    print(f"  T2 Clustering:           {len(multi)} multi-source clusters from 200 articles")
    print(f"  T3 Entity detection:     {len(people)} people detected")
    print(f"  T4 Glossary (seed):      {len(glossary)} entries")
    print(f"  T5 Glossary (clusters):  {len(glossary_cl)} entries")
    print(f"  T6 Validation:           {'PASS' if t6_ok else 'ISSUES'}")
    print()


if __name__ == "__main__":
    main()
