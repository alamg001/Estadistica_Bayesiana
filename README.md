# Estadistica_Bayesiana
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resumen del PDF • Estadística Computacional (UNI)</title>
    <!-- KaTeX CDN para renderizar ecuaciones matemáticas -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
    <style>
        :root {
            --bg: #f8f9fa;
            --text: #212529;
            --accent: #0d6efd;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --bg: #212529;
                --text: #f8f9fa;
                --accent: #4dabf7;
            }
        }
        body {
            font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.7;
            margin: 0;
            padding: 40px 20px;
            max-width: 1100px;
            margin: 0 auto;
        }
        header {
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 3px solid var(--accent);
            padding-bottom: 20px;
        }
        h1 { font-size: 2.4rem; margin: 0 0 10px; color: var(--accent); }
        h2 { font-size: 1.8rem; margin-top: 2.5rem; color: var(--accent); border-left: 6px solid var(--accent); padding-left: 15px; }
        h3 { margin-top: 2rem; color: #495057; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 25px 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        th, td {
            padding: 14px 18px;
            border: 1px solid #dee2e6;
            text-align: left;
            vertical-align: top;
        }
        th {
            background: var(--accent);
            color: white;
            font-weight: 600;
        }
        tr:nth-child(even) { background: rgba(13, 110, 253, 0.05); }
        .katex { font-size: 1.1em; }
        .math { margin: 15px 0; }
        pre {
            background: #f1f3f5;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.95em;
        }
        .github-link {
            text-align: center;
            margin: 50px 0 20px;
            font-size: 1.1em;
        }
        .github-link a {
            background: #212529;
            color: white;
            padding: 12px 24px;
            border-radius: 50px;
            text-decoration: none;
            transition: all 0.3s;
        }
        .github-link a:hover {
            background: #0d6efd;
            transform: translateY(-2px);
        }
        footer { text-align: center; margin-top: 60px; color: #6c757d; font-size: 0.95em; }
    </style>
</head>
<body>
    <header>
        <h1>Resumen completo del PDF</h1>
        <p style="font-size: 1.4rem; margin: 0;"><strong>Estadistica_Bayesiana.pdf</strong> (Computational Statistics)</p>
        <p style="margin-top: 10px; font-size: 1.1rem; opacity: 0.9;">
            Curso de <strong>Estadística Computacional</strong> • Dr. Erick A. Chacón Montalván (UNI)
        </p>
        <p style="max-width: 700px; margin: 25px auto 0; font-size: 1.05rem;">
            Aunque el nombre del archivo dice “Estadística Bayesiana”, el contenido real trata de las 
            <strong>herramientas computacionales necesarias para la inferencia bayesiana</strong> 
            (simulación Monte Carlo, optimización y algoritmos EM).
        </p>
    </header>

    <h2>📌 Estructura del documento</h2>
    <p>El PDF de <strong>554 páginas</strong> se divide en tres módulos principales:</p>
    <ol>
        <li><strong>Generación de números aleatorios</strong> (páginas 1-40)</li>
        <li><strong>Optimización y resolución de ecuaciones no lineales</strong> (páginas 41-72)</li>
        <li><strong>Algoritmo Expectation-Maximization (EM)</strong> (páginas 73-554, con ejemplos y aplicaciones bayesianas)</li>
    </ol>

    <h2>1. Generación de números aleatorios</h2>
    <h3>Definición base (Probability Integral Transform)</h3>
    <div class="math">
        Si \( X \) es una variable aleatoria continua con función de distribución acumulada \( F_X(x) \), entonces
        \[ Y = F_X(X) \sim \text{Uniform}(0,1) \]
    </div>
    <p>Inversamente:</p>
    <div class="math">
        \[ X = F_X^{-1}(U) \quad \text{con } U \sim \text{Uniform}(0,1) \]
    </div>

    <h3>Algoritmos mostrados y para qué sirven</h3>
    <table>
        <thead>
            <tr>
                <th>Algoritmo</th>
                <th>Para qué sirve</th>
                <th>Cómo se implementa (paso a paso)</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Linear Congruential Generator (LCG) y Multiple Recursive Generator (MRG)</td>
                <td>Generar números pseudo-aleatorios Uniform(0,1) de forma determinística y reproducible</td>
                <td>
                    \[ x_i = (a_1 x_{i-1} + \cdots + a_k x_{i-k}) \mod m \]<br>
                    \[ u_i = x_i / m \]
                </td>
            </tr>
            <tr>
                <td>Mersenne Twister / Xoshiro256++</td>
                <td>Generadores modernos con periodo enorme y excelente calidad estadística (usado por defecto en Julia)</td>
                <td>Igual que arriba pero con estructura más compleja (no lineal simple).</td>
            </tr>
            <tr>
                <td>Inverse CDF (Inverse Transform Sampling)</td>
                <td>Generar cualquier distribución cuya CDF inversa sea conocida (Exponential, Pareto, Weibull, etc.)</td>
                <td>
                    1. Generar \( u_i \sim U(0,1) \)<br>
                    2. \( x_i = F^{-1}(u_i) \)
                </td>
            </tr>
            <tr>
                <td>Rejection Sampling</td>
                <td>Generar de distribuciones complicadas cuando no se conoce la inversa de la CDF</td>
                <td>
                    1. Elegir propuesta \( q(x) \) y constante \( M \) tal que \( f(x) \leq M q(x) \)<br>
                    2. Generar \( x \sim q(x) \), \( u \sim U(0,1) \)<br>
                    3. Aceptar si \( u \leq \frac{f(x)}{M q(x)} \)
                </td>
            </tr>
            <tr>
                <td>Composition / Convolution Method</td>
                <td>Generar de mezclas de distribuciones o distribuciones condicionales</td>
                <td>
                    1. Elegir componente \( i \) con probabilidad \( p_i \)<br>
                    2. Generar \( x \) de la distribución del componente \( i \)
                </td>
            </tr>
            <tr>
                <td>Cholesky para vectores normales multivariados</td>
                <td>Generar vectores \( X \sim N_q(\mu, \Sigma) \)</td>
                <td>
                    1. Calcular \( \Sigma = LL^T \) (descomposición de Cholesky)<br>
                    2. Generar \( Z \sim N(0,I) \)<br>
                    3. \( X = \mu + L Z \)
                </td>
            </tr>
        </tbody>
    </table>

    <h2>2. Optimización y resolución de ecuaciones no lineales</h2>
    <h3>Definición base</h3>
    <p>Optimizar \( f(x) \) es equivalente a resolver \( f'(x) = 0 \) (y verificar \( f''(x) > 0 \) para mínimo).</p>

    <h3>Algoritmos mostrados</h3>
    <table>
        <thead>
            <tr>
                <th>Algoritmo</th>
                <th>Para qué sirve</th>
                <th>Implementación paso a paso</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>QR decomposition (Least Squares)</td>
                <td>Resolver mínimos cuadrados lineales de forma numéricamente estable</td>
                <td>
                    \[ X = QR \]<br>
                    \[ \hat{\beta} = R^{-1} (Q^Ty)_{1:p} \]
                </td>
            </tr>
            <tr>
                <td>Linear Programming (Quantile Regression)</td>
                <td>Estimar regresión cuantílica (mínimos absolutos ponderados)</td>
                <td>
                    Reformular como LP:<br>
                    \[ \min p \cdot 1^T r^+ + (1-p)1^T r^- \]<br>
                    sujeto a \( y = X\beta + r^+ - r^- \)
                </td>
            </tr>
            <tr>
                <td>Newton / Newton-Raphson</td>
                <td>Encontrar raíces o máximos de verosimilitud (GLM, MLE)</td>
                <td>
                    \[ x^{(t+1)} = x^{(t)} - \frac{f'(x^{(t)})}{f''(x^{(t)})} \]<br>
                    (o con información de Fisher)
                </td>
            </tr>
            <tr>
                <td>Iteratively Reweighted Least Squares (IRLS)</td>
                <td>Ajuste de Modelos Lineales Generalizados (Bernoulli, Poisson, etc.)</td>
                <td>
                    Es Newton aplicado a la verosimilitud exponencial:<br>
                    \[ \beta^{(t+1)} = (X^T W^{(t)} X)^{-1} X^T W^{(t)} z^{(t)} \]
                </td>
            </tr>
            <tr>
                <td>Steepest Ascent</td>
                <td>Optimización cuando no se quiere calcular la Hessiana completa</td>
                <td>
                    \[ x^{(t+1)} = x^{(t)} + \alpha^{(t)} \nabla l(x^{(t)}) \]
                    (usa solo gradiente)
                </td>
            </tr>
            <tr>
                <td>Gauss-Newton</td>
                <td>Mínimos cuadrados no lineales (curvas de crecimiento, modelos no lineales)</td>
                <td>Aproxima la función no lineal \( h(x_i,\theta) \) por Taylor lineal y resuelve LS ordinario en cada iteración (no necesita Hessiana completa)</td>
            </tr>
        </tbody>
    </table>

    <h2>3. Expectation-Maximization (EM) Algorithm</h2>
    <h3>Definición</h3>
    <p>Cuando la verosimilitud completa es difícil de maximizar directamente (datos faltantes, variables latentes, mezclas gaussianas, modelos bayesianos con posterior no conjugada), se usa EM.</p>

    <h3>Pasos del algoritmo EM (teoría + implementación)</h3>
    <ol>
        <li><strong>E-step</strong>: Calcular la esperanza de la log-verosimilitud completa condicionada a los datos observados y al parámetro actual:
            \[ Q(\theta \mid \theta^{(t)}) = \mathbb{E}_{\theta^{(t)}}[\log L(\theta; y_{\text{obs}}, y_{\text{lat}})] \]
        </li>
        <li><strong>M-step</strong>: Maximizar \( Q \) respecto a \( \theta \):
            \[ \theta^{(t+1)} = \arg\max_\theta Q(\theta \mid \theta^{(t)}) \]
        </li>
        <li>Repetir hasta convergencia.</li>
    </ol>
    <p>El PDF muestra ejemplos clásicos (mezclas gaussianas, modelos con datos faltantes, etc.) y cómo se relaciona con inferencia bayesiana (cuando se usa como aproximación al posterior).</p>

    <div class="github-link">
        <a href="https://github.com" target="_blank" style="display: inline-block;">
            📥 Descargar como archivo HTML → Guardar como <strong>resumen-estadistica-computacional.html</strong>
        </a>
    </div>

    <footer>
        Resumen preparado para GitHub • Basado en el PDF de la Universidad Nacional de Ingeniería (UNI) • Marzo 2026
    </footer>

    <script>
        // Renderizar automáticamente todas las ecuaciones KaTeX
        renderMathInElement(document.body, {
            delimiters: [
                { left: "$$", right: "$$", display: true },
                { left: "$", right: "$", display: false }
            ]
        });
    </script>
</body>
</html>
