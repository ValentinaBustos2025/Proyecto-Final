# Plantilla de tesis - Departamento de Física, Universidad de los Andes

Esta plantilla fue reconstruida a partir del PDF de referencia suministrado por la usuaria, manteniendo su maquetación principal.

## Formato reproducido

- Papel: Letter (612 x 792 pt).
- Clase: `article`, 10 pt.
- Fuente: Computer Modern / CM-Super con codificación T1.
- Márgenes: 0.85 in en los cuatro lados.
- Pie de página: `\footskip = 20 pt`.
- Texto: 10 pt, interlínea estándar de LaTeX (~12 pt).
- Sangría de primera línea: la predeterminada de LaTeX (~15 pt), incluida después de títulos mediante `indentfirst`.
- Sección: 14.4 pt, negrita.
- Subsección: 12 pt, negrita.
- Subsubsección: 10 pt, negrita.
- Portada, dedicatoria, agradecimientos e índice ajustados a las posiciones del PDF de referencia.
- Numeración de secciones con punto final (`1.`, `2.1.`, etc.) mediante `babel` en español.
- El índice conserva los números sin punto final, como en el PDF original.

## Cómo usarla en Overleaf

1. Sube el ZIP como un proyecto nuevo.
2. Verifica que el archivo principal sea `main.tex`.
3. Edita los datos personales y el título en `config.tex`.
4. Sustituye el contenido de `frontmatter/agradecimientos.tex` y `frontmatter/dedicatoria.tex`.
5. Escribe los capítulos dentro de `capitulos/`.
6. Añade referencias a `referencias.bib`.

## Importante para conservar la apariencia

No añadas paquetes que cambien la fuente, como `lmodern`, `times`, `newtxtext`, `mathptmx`, `libertinus` o similares. La fuente del PDF de referencia corresponde al aspecto clásico de Computer Modern/CM-Super.

El salto de línea del título de portada se controla manualmente en `config.tex`, porque el documento de referencia también compone el título en dos líneas centradas.
