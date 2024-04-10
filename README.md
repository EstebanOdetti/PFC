# Desarrollo de arquitecturas basadas en redes neuronales artificiales para aproximar la solución de las ecuaciones de transferencia de calor y dinámica estructural.

![Licencia](https://img.shields.io/badge/license-MIT-green) ![Python](https://img.shields.io/badge/made%20with-Python-blue.svg) ![Versión](https://img.shields.io/badge/version-1.0.0-blue)

## Resumen 📄

Hasta hace unos años se simulaban problemas de tipo estático, pero en la actualidad los cálculos dinámicos resultan más representativos, aunque incrementan exponencialmente el costo computacional. En este contexto, la aplicación de técnicas de Machine Learning pueden resultar una alternativa viable para reducir dichos costos. Se propone estudiar y desarrollar algoritmos basados en redes neuronales artificiales que resuelvan, primero un problema de transferencia de calor clásico (resuelto hasta ahora por
Elementos Finitos), para luego extrapolarlo a la mecánica estructural. El proceso de aplicación de Machine Learning en dinámica estructural consiste en ajustar un modelo para estimar de manera aproximada la tensión resultante y la frecuencia predominante de una bicicleta sometida a excitaciones cuasi-aleatorias. Aunque la bicicleta es un sistema sencillo comparado con otras maquinarias, por ejemplo la agrícola, su estudio nos brinda la oportunidad perfecta para probar y perfeccionar nuestras redes neuronales en un contexto más controlable y comprensible. Este proyecto nos permitió no solo analizar y ajustar nuestros métodos de simulación y análisis, sino también obtener conclusiones valiosas sobre el funcionamiento y la eficacia de las redes neuronales. Estas estarán diseñadas para estimar los valores de la tensión resultante medida y frecuencia predominante en dos puntos de una bicicleta de aluminio.

### Objetivo General 🎯
Desarrollar una arquitectura de redes neuronales capaz de retornar una aproximación para la ecuación de calor y evaluar su aplicación en sistemas mecánicos simples para estimar la tensión resultante y la frecuencia predominante.

## Alcance 🔍

El alcance de este proyecto es explorar la viabilidad de aplicar técnicas de aprendizaje automático (Machine Learning) para predecir soluciones a la ecuación de calor mediante el uso de redes neuronales. Además, se busca estimar la tensión resultante y la frecuencia predominante en dinámica estructural.

### Conclusiones generales ✅

En el ámbito de la ingeniería y la física aplicada, la resolución de ecuaciones diferenciales desempeña un papel fundamental en la comprensión y modelado de fenómenos complejos. Abordar la solución de ecuaciones diferenciales, en particular, para aproximarse a la ecuación general del transporte de calor y la tensión-deformación, representa un desafío esencial. Estos problemas son crucialmente relevantes para el diseño y análisis de sistemas y estructuras en diversas disciplinas ingenieriles. En este contexto, hemos obtenido resultados prometedores, aunque es imperativo reconocer que se realizaron ciertas simplificaciones. A medida que los casos a considerar se tornan más complejos y el dominio de aplicación se expande, se requiere una dedicación más intensiva
para adaptarse a nuevos desafíos que puedan surgir. Pese a esto, los resultados obtenidos son consistentes con nuestras expectativas y se alinean con los objetivos planteados para un proyecto de naturaleza exploratoria. Aunque el nivel de rigurosidad necesario para abordar íntegramente estas cuestiones de gran complejidad supera el alcance de nuestra investigación, hemos proporcionado un punto de inicio para futuras investigaciones. 
Concluimos que las redes neuronales artificiales tienen la capacidad de aproximar soluciones a una amplia gama de ecuaciones diferenciales. Aunque nuestro proyecto demostró la capacidad de aproximación con un margen de error, es crucial reconocer las limitaciones impuestas por el tiempo y los recursos utilizados. Se evidenció que la aproximación difiere de las soluciones proporcionadas por métodos numéricos clásicos. Cada experiencia implicó la realización de mediciones directas en el terreno mediante el uso de una bicicleta, prefiriendo la captura de datos reales mediante acelerómetros en vez de optar por la simulación de esta señal. Esta decisión, junto a las restricciones temporales del proyecto, limitó el acceso a un amplio conjunto de datos. Esta condición incide en la capacidad de aprendizaje del modelo. En última instancia, este trabajo representa un paso inicial hacia la contribución en el cruce entre la resolución de ecuaciones diferenciales y el aprendizaje automático. Se identifica el potencial de las redes neuronales para abordar estas complejas problemáticas, pero se reconoce la necesidad de investigaciones futuras para perfeccionar y ampliar esta aproximación.

## Herramientas 🛠️

* Octave 6.2
* Visual Studio Code v1.88
* Python 3.11 de 64 bits
* pytorch
* matplotlib
* numpy
* scipy
* sklearn
* pandas
* itertools
* torch_geometric
* Acelerometer Meter (aplicación disponible en la tienda de Android)
* SolidWorks 2023
* Vibration Data Toolbox

## Autores ✒️

* **ODETTI, Esteban Jorge y POZZER, Andrés Fernando** - *PFC* - [EstebanOdetti ]([URL](https://github.com/EstebanOdetti)) [AndresPoz]([URL](https://github.com/EstebanOdetti](https://github.com/AndresPoz)))

También puedes mirar la lista de todos los [contribuyentes](URL) que han participado en este proyecto.

## Licencia 📄

Este proyecto está bajo Licencia MIT - mira el archivo [LICENSE.md](URL) para detalles.

## Gracias por leer!🤓
