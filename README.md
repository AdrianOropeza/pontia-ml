# Recordar

company (94% nulos) → la convertiste en binaria has_company porque el 94% vacío no aportaba y lo que importaba era "¿hay empresa sí/no?".

agent (14% nulos, 333 IDs distintos) → misma lógica, binaria has_agent, porque 333 agentes no son aprendibles.

children (4 nulos, numérica) → rellenaste con 0, la opción conservadora que no inventa valores altos.

country (488 nulos, texto) → rellenaste con "Unknown", honesto, sin falsear procedencias.


vimos un desbalanceo del 63/37, moderado pero suficiente para que la accuracy no baste, así que priorizamos F1/AUC-ROC". Coménta esto con B y C, porque la elección de métrica es decisión de equipo (afecta a cómo C evalúa y selecciona el mejor modelo).