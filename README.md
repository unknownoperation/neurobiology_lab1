# Анализ
В работе была реализована модель Ходжкина-Хаксли, описывающая генерацию и распространение потенциала действия в клеточной мембране нейрона. В первой симуляции подаваемый на мембрану ток задавали четырьмя значениями: 20,20,32 и 47 нА с длительностью подачи 30,50,100 и 100 мс соответственно, временной промежуток между импульсами составлял 50 мс. 
При подаче тока на мембрану происходит увеличение её потенциала с потенциала покоя (около - 65 мВ) до порогового значения. При достижении порогового значения мы наблюдаем пик (около 40 мВ), после чего в отсутствие тока мембранный потенциал уменьшается до состояния ниже потенциала покоя с последующей экспоненциальной релаксацией до состояния покоя. Так как подаваемый на мембрану ток в симуляции продолжительный, после первого пика мы наблюдаем не релаксацию, а генерацию последующих пиков, но уже с меньшей амплитудой. Так как подаваемый ток постоянный, то частота генерации пиков одинакова. Из графиков видно, что при увеличении длительности подачи импульса увеличивается количество пиков, а при увеличении величины тока увеличивается пороговое значение потенциала на мембране.
Второй график иллюстрирует динамику переменных m, h и n, которые показывают динамику активации и инактивации натриевых каналов и активации калиевых каналов соответственно. При подаче тока и увеличении мембранного потенциала, переменная m растёт скачком до единицы, а h уменьшается с 0,6 почти до нуля, что означает открытие натриевых каналов и движение ионов натрия в клетку. Входящий ток ионов увеличивает потенциал клетки, что вызывает инактивацию натриевых каналов, и h увеличивается. Это уменьшает потенциал и активирует калиевые каналы, n растёт до единицы, потенциал уменьшается до отрицательных значений.

Так как модель Ходжкина-Хаксли описывается дифференциальными уравнениями с четырьмя переменными, которые зависят от времени, удобнее визуализировать две переменные в фазовом пространстве — потенциал мембраны и переменные ионных каналов. Из графиков видно, что мембранный потенциал циклически изменяется с изменением открытия ионных каналов (например, потенциал увеличивается с активацией натриевых и калиевых каналов).
(см. графики в папке simulation1)

Во второй симуляции ток задавался значениями 150 и 100 длительностью 1 мс каждый с интервалами между импульсами 1 мс. Так как импульсы подавались непрерывно в течение всей симуляции, и интервал между ними был настолько мал, что потенциал мембраны не успевал релаксировать, мы видим на графике зависимости потенциала от времени непрерывные осцилляции одинаковой амплитуды и одинаковой частотой. Небольшие пики (до -40 мВ) соответствуют времени перерыва подачи тока
(см. графики в папке simulation2)
