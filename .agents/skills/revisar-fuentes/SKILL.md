---
name: revisar-fuentes
description: Revisar que cada afirmación se encuentre respaldada con datos. Se utiliza después de cada afirmación nueva, dependiente de datos, que sea agregada al informe.
---
1. Se lee el texto completamente y se identifican todas las afirmaciones que se desprendan a partir de los datos. Se marcan como "PENDIENTE".
2. Se revisan todas las afirmaciones identificadas buscando su dato de origen y recalculando el valor obtenido utilizando el script `recalcular_valores.py` ubicado en `scripts/recalcular_valores.py` a partir de la información provista en `smart_city_traffic_mobility.csv`.
3. Se compara el valor exhibido en el informe con el calculado. Si existen diferencias entonces se marca dicha afirmación como "REVISAR". Si no existen diferencias se marca como "APROBADO".
4. Finalmente se entrega un archivo .csv en `resultados/tablas/` donde se resumen las afirmaciones realizadas, dónde se encuentran en el informe, si su estado ("PENDIENTE", "REVISAR", "APROBADO")