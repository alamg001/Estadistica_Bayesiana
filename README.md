<div align="center">

<img src="https://www.uni.edu.pe/images/logo_uni.png" width="75"/>

# Estadística Computacional

**Doctorando:** Antonio Lam García  
**Profesor:** Dr. Erick A. Chacón Montalván

Escuela Profesional de Ingeniería Estadística  
Facultad de Ingeniería Económica, Estadística y Ciencias Sociales  
**Universidad Nacional de Ingeniería — Lima, Perú**

---

*Notas del curso `Estadistica_Bayesiana.pdf` · 554 páginas*

</div>

---

## Sobre este documento

Este repositorio recoge las notas de un curso de **Estadística Computacional** impartido en el programa de posgrado de la UNI. A pesar del nombre del archivo, el contenido no versa sobre inferencia bayesiana en abstracto, sino sobre el conjunto de herramientas algorítmicas que la hacen computacionalmente viable: generación de muestras de distribuciones arbitrarias, optimización numérica, y estimación bajo datos incompletos mediante el algoritmo EM. Estas piezas forman el sustrato técnico sobre el que descansan los métodos MCMC y la inferencia bayesiana moderna.

---

## Índice de módulos

| # | Módulo | Páginas | Núcleo conceptual |
|---|--------|---------|-------------------|
| 1 | [Generación de números aleatorios](#1-generación-de-números-aleatorios) | 1–40 | PIT; RNGs; inverse CDF; rejection sampling; Cholesky |
| 2 | [Optimización y ecuaciones no lineales](#2-optimización-y-resolución-de-ecuaciones-no-lineales) | 41–72 | QR; LP cuantílico; Newton; IRLS; Gauss-Newton |
| 3 | [Algoritmo EM](#3-algoritmo-expectation-maximization-em) | 73–200 | Verosimilitud completa; E-step/M-step; ascenso monótono |
| 4 | [Métodos de Monte Carlo](#4-métodos-de-monte-carlo) | 200–300 | Estimador MC; importance sampling; SIR; control variates |
| 5 | [MCMC: Gibbs y Metropolis-Hastings](#5-markov-chain-monte-carlo-mcmc) | 300–554 | Cadenas de Markov; balance detallado; Gibbs; MH |

---

## 1. Generación de Números Aleatorios

### 1.1 Fundamento: Transformada Integral de Probabilidad

> **Proposición (Probability Integral Transform).** Sea $X$ variable aleatoria continua con CDF $F_X$. Entonces:
>
> $$Y := F_X(X) \;\sim\; \mathrm{Uniform}(0,1)$$
>
> Inversamente, si $U \sim \mathrm{Uniform}(0,1)$, entonces $X := F_X^{-1}(U)$ tiene distribución $F_X$.

**Demostración.** Para $y \in (0,1)$:

$$F_Y(y) \;=\; \Pr(Y \leq y) \;=\; \Pr\!\bigl(F_X(X) \leq y\bigr) \;=\; \Pr\!\bigl(X \leq F_X^{-1}(y)\bigr) \;=\; y. \quad \blacksquare$$

---

### 1.2 Generadores de Números Uniformes

#### 1.2.1 Estructura de un RNG determinista

Un RNG queda caracterizado por la quintupla $(S,\,\mu,\,f,\,U,\,g)$: espacio de estados $S$, distribución inicial $\mu$ (*seed*), función de transición $f:S\to S$, espacio de salida $U$ y función de output $g:S\to U$. El **período** es el menor $j>0$ tal que $s_{i+j} = s_i$.

#### 1.2.2 Recurrencia lineal: LCG y MRG

$$x_i \;=\; \bigl(a_1\,x_{i-1} + \cdots + a_k\,x_{i-k}\bigr)\;\bmod\; m, \qquad u_i \;=\; x_i\,/\,m.$$

| Generador | Parámetros | Período máximo |
|-----------|-----------|----------------|
| **LCG** ($k=1$) | $x_i = a\,x_{i-1}\bmod m$ | $m-1$ |
| **MRG** ($k>1$, $m$ primo) | $a_1,\ldots,a_k \in \mathbb{Z}_m$ | $m^k - 1$ |

Validación empírica (TestU01): $H_0: u_i \overset{\mathrm{iid}}{\sim} U(0,1)$. Se rechaza si el $p$-valor es extremadamente pequeño o extremadamente grande.

#### 1.2.3 Generadores modernos

**Mersenne Twister** y **Xoshiro256++** (por defecto en Julia) ofrecen períodos del orden de $2^{19937}-1$ y excelente equidistribución en alta dimensión, mediante transformaciones de estado no lineales sobre vectores de bits.

---

### 1.3 Métodos de Muestreo No Uniforme

#### Algoritmo 1 — Inverse CDF (Transformada Inversa)

```
Entrada : F^{-1} conocida analíticamente, tamaño m
Para i = 1, ..., m:
  1. Generar  u_i ~ Uniform(0,1)
  2. Devolver x_i = F^{-1}(u_i)
```

**Ejemplos:**

| Distribución | $F^{-1}(u)$ |
|---|---|
| $\mathrm{Exp}(\theta)$ | $-\theta\ln(1-u)$ |
| $\mathrm{Pareto}(\lambda,k)$ | $\lambda\,(1-u)^{-1/k}$ |
| $\mathrm{Weibull}(\lambda,k)$ | $\lambda\,\bigl(-\ln(1-u)\bigr)^{1/k}$ |

---

#### Algoritmo 2 — Rejection Sampling

Sea $\pi(x) = k\,J(x)$ la densidad objetivo con constante normalizante $k$ desconocida.

```
Entrada : J(x), propuesta q(x), cota M tal que J(x) <= M * q(x)
Repetir:
  1. Generar  x ~ q(x)
  2. Generar  u ~ Uniform(0,1)
  3. Si  u <= J(x) / (M * q(x))  -->  aceptar x
     En caso contrario            -->  rechazar, repetir
```

**Corrección:**

$$\Pr(x \mid I=1) \;=\; \frac{q(x)\cdot\dfrac{J(x)}{Mq(x)}}{\int q(x)\dfrac{J(x)}{Mq(x)}\,dx} \;=\; \frac{J(x)}{\int J(x)\,dx} \;=\; \pi(x). \quad \blacksquare$$

Eficiencia: $\Pr(I=1) = 1/(kM)$. Un $M$ ajustado reduce el costo computacional.

---

#### Algoritmo 3 — Composición / Convolución

Para $\pi(x) = \sum_{i=1}^{k} p_i\,\pi_i(x)$ o $\pi(x) = \int \pi(y)\,\pi(x\mid y)\,dy$:

```
1. Generar  I = i  con probabilidad p_i   (o generar y ~ pi(y))
2. Generar  x ~ pi_i(x)                  (o generar x ~ pi(x|y))
3. Devolver x
```

---

#### Algoritmo 4 — Normal Multivariada vía Cholesky

Sea $X \sim N_q(\mu, \Sigma)$ con $\Sigma = LL^\top$.

```
1. Calcular  L  tal que  Sigma = L * L^T   (factorizacion de Cholesky)
2. Generar   Z ~ N(0, I_q)
3. Devolver  X = mu + L * Z
```

**Verificación:** $\mathbb{E}[X]=\mu$, $\mathrm{Cov}(X)=L\,I_q\,L^\top=\Sigma$. $\blacksquare$

---

## 2. Optimización y Resolución de Ecuaciones No Lineales

### 2.1 Marco General

Optimizar $f:\mathcal{A}\to\mathbb{R}$ se reduce a resolver $f'(x_0)=0$ verificando condiciones de segundo orden. El vínculo estadístico es directo: el MLE satisface $S(\hat\theta)=\nabla_\theta\,\ell(\hat\theta)=0$, donde $S(\theta)$ es la función score.

---

### 2.2 Mínimos Cuadrados vía Descomposición QR

Dado $y = X\beta + \varepsilon$, la solución directa $\hat\beta=(X^\top X)^{-1}X^\top y$ es numéricamente inestable. Se usa $X = Q_f R$:

$$\mathcal{L}(\beta) = \|y - X\beta\|^2 = \|f - R\beta\|^2 + \|r\|^2,$$

minimizada resolviendo el sistema triangular $R\hat\beta = f$. Coste $O(p^2)$; condicionamiento óptimo.

---

### 2.3 Regresión Cuantílica como Programa Lineal

La estimación del cuantil condicional $Q_p(Y_i\mid x_i)=x_i^\top\beta_p$ minimiza la función de pérdida:

$$\hat\beta_p \;=\; \arg\min_{\beta_p} \left\lbrace \sum_{i:\;y_i \geq x_i^\top\beta_p} p\,\bigl|y_i - x_i^\top\beta_p\bigr| \;+\; \sum_{i:\;y_i < x_i^\top\beta_p} (1-p)\,\bigl|y_i - x_i^\top\beta_p\bigr| \right\rbrace.$$

Con $e = r^{+} - r^{-}$, $r^{+}=\max(0,e)$, $r^{-}=\max(0,-e)$, se convierte en un **LP** estándar:

$$\min_{r^{+},\,r^{-},\,\beta_p}\; p\,\mathbf{1}^\top r^{+} + (1-p)\,\mathbf{1}^\top r^{-} \quad \text{s.a.} \quad y = X\beta_p + r^{+} - r^{-},\;\; r^{+},r^{-} \geq 0.$$

---

### 2.4 Métodos Iterativos para Programación No Lineal

#### Algoritmo 5 — Newton-Raphson

```
Inicializar theta^(0)
Para t = 0, 1, 2, ...:
  1. Calcular  f'(theta^(t))  y  f''(theta^(t))
  2. Actualizar:
       theta^(t+1) = theta^(t) - f'(theta^(t)) / f''(theta^(t))
  3. Parar si  |theta^(t+1) - theta^(t)| < epsilon
```

En $\mathbb{R}^p$:

$$\theta^{(t+1)} = \theta^{(t)} - \bigl[\nabla^2\ell\bigl(\theta^{(t)}\bigr)\bigr]^{-1}\,\nabla\ell\bigl(\theta^{(t)}\bigr).$$

Usar la **información de Fisher esperada** $\mathcal{I}_E(\theta)=-\mathbb{E}[\nabla^2\ell(\theta)]$ en lugar de $-\nabla^2\ell$ da el **Fisher scoring**.

---

#### Algoritmo 6 — IRLS (GLMs)

Para GLM Bernoulli-logit, $\theta_i=\mathrm{logit}(\pi_i)=x_i^\top\beta$:

$$\nabla\ell(\beta) = X^\top(y-\pi), \qquad \nabla^2\ell(\beta) = -X^\top W X, \qquad W = \mathrm{diag}\bigl\{\pi_i(1-\pi_i)\bigr\}.$$

```
Inicializar beta^(0)
Para t = 0, 1, 2, ...:
  1. Calcular  pi^(t)  = sigmoid(X * beta^(t))
  2. Calcular  W^(t)   = diag{ pi_i^(t) * (1 - pi_i^(t)) }
  3. Calcular  z^(t)   = X*beta^(t) + (W^(t))^{-1} * (y - pi^(t))
  4. Resolver WLS:
       beta^(t+1) = (X^T * W^(t) * X)^{-1} * X^T * W^(t) * z^(t)
  5. Parar si  ||beta^(t+1) - beta^(t)|| < epsilon
```

---

#### Algoritmo 7 — Steepest Ascent

```
Inicializar x^(0), secuencias {alpha^(t)} decrecientes
Para t = 0, 1, 2, ...:
  1. Calcular gradiente  grad_l(x^(t))
  2. Actualizar:
       x^(t+1) = x^(t) + alpha^(t) * grad_l(x^(t))
  3. Parar si  ||grad_l(x^(t+1))|| < epsilon
```

$$x^{(t+1)} = x^{(t)} + \alpha^{(t)}\,\nabla\ell\bigl(x^{(t)}\bigr).$$

Equivale al método Newton con $M^{(t)}=-I$ (sin Hessiana).

---

#### Algoritmo 8 — Gauss-Newton (LS No Lineal)

Para $Y_i = h(x_i,\theta)+\varepsilon_i$, linealizar $h$ por Taylor alrededor de $\theta^{(t)}$:

```
Inicializar theta^(0)
Para t = 0, 1, 2, ...:
  1. Residuales: z_i^(t) = y_i - h(x_i, theta^(t))
  2. Jacobiano:  A_i^(t) = h'(x_i, theta^(t))
  3. Resolver LS lineal:
       theta^(t+1) = theta^(t) + (A^T * A)^{-1} * A^T * z^(t)
  4. Parar si  ||theta^(t+1) - theta^(t)|| < epsilon
```

No requiere la Hessiana completa $\nabla^2\mathcal{L}$; sólo el Jacobiano $A^{(t)}$.

---

## 3. Algoritmo Expectation-Maximization (EM)

### 3.1 Motivación

Sea $y_{\mathrm{obs}}$ los datos observados y $y_{\mathrm{lat}}$ los datos latentes. La verosimilitud marginal

$$L(\theta;\,y_{\mathrm{obs}}) = \int L(\theta;\,y_{\mathrm{obs}},y_{\mathrm{lat}})\,f(y_{\mathrm{lat}}\mid y_{\mathrm{obs}};\theta)\,dy_{\mathrm{lat}}$$

es generalmente intratable. La **verosimilitud completa** $\ell(\theta;\,y_{\mathrm{obs}},y_{\mathrm{lat}})$ suele ser manejable (familia exponencial).

---

### 3.2 El Algoritmo EM

Dado el estimado actual $\theta^{(t)}$, definir la función auxiliar:

$$Q\bigl(\theta,\,\theta^{(t)}\bigr) = \mathbb{E}_{\theta^{(t)}}\!\bigl[\ell(\theta;\,y_{\mathrm{obs}},Y_{\mathrm{lat}})\mid y_{\mathrm{obs}}\bigr].$$

#### Algoritmo 9 — EM

```
Inicializar  theta^(0)
Para t = 0, 1, 2, ...:

  -- E-step --------------------------------------------------
  Calcular Q(theta, theta^(t)) como funcion de theta,
  integrando ell(theta; y_obs, y_lat) respecto a
  f(y_lat | y_obs; theta^(t))

  -- M-step --------------------------------------------------
  theta^(t+1) = argmax_theta  Q(theta, theta^(t))

  -- Criterio de parada --------------------------------------
  Parar si  |ell(theta^(t+1); y_obs) - ell(theta^(t); y_obs)| < epsilon
```

---

### 3.3 Convergencia: Ascenso Monótono Garantizado

> **Teorema.** En cada iteración del EM: $\;\ell(\theta^{(t+1)};\,y_{\mathrm{obs}}) \;\geq\; \ell(\theta^{(t)};\,y_{\mathrm{obs}})$.

**Demostración.** Descomponer la log-verosimilitud observada:

$$\ell(\theta;\,y_{\mathrm{obs}}) = Q(\theta,\,\theta^{(t)}) - H(\theta,\,\theta^{(t)}),$$

donde $H(\theta,\theta^{(t)}) = \mathbb{E}_{\theta^{(t)}}\!\bigl[\log f(Y_{\mathrm{lat}}\mid y_{\mathrm{obs}};\theta)\bigr]$.

Por la desigualdad de Jensen aplicada a $-\log$:

$$H(\theta^{(t+1)},\theta^{(t)}) - H(\theta^{(t)},\theta^{(t)}) = \mathbb{E}_{\theta^{(t)}}\!\left[\log\frac{f(Y_{\mathrm{lat}}\mid y_{\mathrm{obs}};\theta^{(t+1)})}{f(Y_{\mathrm{lat}}\mid y_{\mathrm{obs}};\theta^{(t)})}\right] \leq 0.$$

El paso M garantiza $Q(\theta^{(t+1)},\theta^{(t)})\geq Q(\theta^{(t)},\theta^{(t)})$, por lo que:

$$\ell(\theta^{(t+1)}) - \ell(\theta^{(t)}) = \underbrace{\bigl[Q(\theta^{(t+1)},\theta^{(t)})-Q(\theta^{(t)},\theta^{(t)})\bigr]}_{\geq\,0} - \underbrace{\bigl[H(\theta^{(t+1)},\theta^{(t)})-H(\theta^{(t)},\theta^{(t)})\bigr]}_{\leq\,0} \;\geq\; 0. \quad \blacksquare$$

---

### 3.4 Errores Estándar: Descomposición de Información

$$\underbrace{-\nabla^{2}_{\theta}\log f(y_{\mathrm{obs}};\theta)}_{\text{informacion observada}} = \underbrace{-\nabla^{2}_{\theta}\,Q(\theta,\theta^{(t)})}_{\text{informacion completa}} \;-\; \underbrace{\Bigl(-\nabla^{2}_{\theta}\,H(\theta,\theta^{(t)})\Bigr)}_{\text{informacion faltante}},$$

evaluada en $\hat\theta$. La información faltante se expresa como:

$$-\nabla^{2}_{\theta}\,H(\theta,\theta^{(t)})\Big|_{\hat\theta} = \mathrm{Var}\!\left[\nabla_\theta\log f(y_{\mathrm{lat}},y_{\mathrm{obs}};\theta)\mid y_{\mathrm{obs}},\hat\theta\right].$$

---

### 3.5 Ejemplo: Mezcla de Gaussianas

**Modelo:** $Y \sim \pi_1\,N(\mu_1,\sigma_1^{2}) + \pi_2\,N(\mu_2,\sigma_2^{2})$, con indicador latente $Z_i\in\{1,2\}$.

**E-step** — Responsabilidades posteriores:

$$\gamma_{ij} := \Pr(Z_i=j\mid y_i) = \frac{\pi_j\,\phi\!\bigl(y_i;\,\mu_j^{(t)},\,\sigma_j^{(t)\,2}\bigr)}{\sum_{k=1}^{2}\pi_k\,\phi\!\bigl(y_i;\,\mu_k^{(t)},\,\sigma_k^{(t)\,2}\bigr)}.$$

**M-step** — Actualizaciones en forma cerrada:

$$\hat\mu_j = \frac{\sum_i y_i\,\gamma_{ij}}{\sum_i \gamma_{ij}}, \qquad \hat\sigma_j^{2} = \frac{\sum_i (y_i-\hat\mu_j)^2\,\gamma_{ij}}{\sum_i \gamma_{ij}}, \qquad \hat\pi_j = \frac{1}{n}\sum_i \gamma_{ij}.$$

---

### 3.6 Ejemplo: Datos Censurados — Gamma$(2,\delta)$

Con $m$ observaciones completas y $n-m$ censuradas a la derecha en $a$:

$$\mathbb{E}[Z_i\mid Z_i>a] = \frac{2 + 2a\delta + a^{2}\delta^{2}}{\delta\,(1+a\delta)}.$$

La actualización del paso M es:

$$\delta^{(t+1)} = \frac{2n}{\displaystyle\sum_{i=1}^{m}y_i \;+\; (n-m)\cdot\dfrac{2+2a\delta^{(t)}+a^{2}\bigl(\delta^{(t)}\bigr)^{2}}{\delta^{(t)}\bigl(1+a\delta^{(t)}\bigr)}}.$$

---

## 4. Métodos de Monte Carlo

### 4.1 Estimador de Monte Carlo

Toda cantidad expresable como esperanza bajo $f$:

$$\mu = \int h(x)\,f(x)\,dx = \mathbb{E}_{f}[h(X)],$$

se estima con:

$$\hat\mu_{\mathrm{MC}} = \frac{1}{m}\sum_{i=1}^{m}h(X_i), \qquad X_i \overset{\mathrm{iid}}{\sim} f.$$

Consistencia por la LLGN; distribución asintótica por el TCL:

$$\sqrt{m}\,\bigl(\hat\mu_{\mathrm{MC}}-\mu\bigr) \;\xrightarrow{d}\; N\!\bigl(0,\,\mathrm{Var}[h(X)]\bigr).$$

---

### 4.2 Importance Sampling

Densidad instrumental $g$ con $\mathrm{supp}(h\cdot f)\subseteq\mathrm{supp}(g)$. Pesos:

$$w^{*}(x) = \frac{f(x)}{g(x)}, \qquad w(x_i) = \frac{w^{*}(x_i)}{\sum_{j=1}^{m}w^{*}(x_j)}.$$



<p align="center">
  <img src="https://github.com/alamg001/alamg001-CicloII.github.io/blob/main/Muestreo%20por%20Importancia%20Normalizado.png?raw=true" alt="Muestreo por Importancia Normalizado" width="500"/>
</p>



| Estimador | Fórmula | Sesgo |
|---|---|---|
| No sesgado | $\hat\mu_{\mathrm{IS}}^{*} = \dfrac{1}{m}\sum_i h(X_i)\,w^{*}(X_i)$ | $0$ |
| De razón | $\hat\mu_{\mathrm{IS}} = \sum_i h(X_i)\,w(X_i)$ | $O(1/m)$ pero menor varianza |

$g$ debe tener **colas más pesadas** que $f$ para estabilidad de los pesos.

---

### 4.3 Sampling Importance Resampling (SIR)

#### Algoritmo 10 — SIR

```
Entrada : densidad objetivo pi(x), propuesta g(x), m >> n
1. Generar candidatos  y_1, ..., y_m  ~  g(·)
2. Calcular pesos:   w*(y_i) = pi(y_i) / g(y_i)
3. Normalizar:        w(y_i) = w*(y_i) / sum_j w*(y_j)
4. Remuestrear  x_1,...,x_n  de {y_i} con probabilidades {w(y_i)}
Salida: muestra aproximada de pi(x),  condicion: n/m --> 0
```

---

### 4.4 Control Variates

Si se conoce $\vartheta=\mathbb{E}[c(Y)]$ exactamente:

$$\hat\mu_{\mathrm{CV}} = \hat\mu_{\mathrm{MC}} + \lambda\,(\hat\vartheta_{\mathrm{MC}}-\vartheta),$$

con $\lambda_{\mathrm{opt}} = -\mathrm{Cov}[\hat\mu_{\mathrm{MC}},\hat\vartheta_{\mathrm{MC}}]\,/\,\mathrm{Var}[\hat\vartheta_{\mathrm{MC}}]$. La reducción de varianza es:

$$\mathrm{Var}[\hat\mu_{\mathrm{CV}}] = \mathrm{Var}[\hat\mu_{\mathrm{MC}}] - \frac{\mathrm{Cov}[\hat\mu_{\mathrm{MC}},\hat\vartheta_{\mathrm{MC}}]^{2}}{\mathrm{Var}[\hat\vartheta_{\mathrm{MC}}]} \;\leq\; \mathrm{Var}[\hat\mu_{\mathrm{MC}}].$$

---

## 5. Markov Chain Monte Carlo (MCMC)

### 5.1 Cadenas de Markov: Propiedades Clave

| Propiedad | Definición |
|---|---|
| **Irreducible** | $\exists\,n: \Pr(X_n=j\mid X_0=i)>0$ para todo $i,j$ |
| **Aperiódica** | $\gcd\{n:\Pr(X_n=j\mid X_0=i)>0\}=1$ |
| **Positiva recurrente** | $\exists$ única $\pi$ estacionaria: $\pi P=\pi$ |

**Balance detallado** (condición suficiente para estacionariedad):

$$\pi(x)\,P(x,y) = \pi(y)\,P(y,x) \quad \forall\,x,y.$$

---

### 5.2 Muestreo de Gibbs

**Objetivo:** muestrear de $\pi(x_1,\ldots,x_d)$ conociendo sólo las condicionales completas.

#### Algoritmo 11 — Gibbs Sampling

```
Inicializar  x^(0) = (x_1^(0), ..., x_d^(0))
Para t = 1, 2, ..., m:
  x_1^(t) ~ pi(X_1 | x_2^(t-1), ..., x_d^(t-1))
  x_2^(t) ~ pi(X_2 | x_1^(t),   x_3^(t-1), ..., x_d^(t-1))
     ...
  x_d^(t) ~ pi(X_d | x_1^(t),   ..., x_{d-1}^(t))
Descartar burn-in de longitud l
Salida: {x^(l+1), ..., x^(m)}  ~  pi(x)
```

Gibbs es un caso especial de MH con $\alpha\equiv 1$.

**Ejemplo — Modelo Normal bayesiano** con priors semi-conjugados $\mu\sim N(\mu_0,\sigma_0^2)$, $\sigma^2\sim\mathrm{IGamma}(a,b)$:

$$\mu\mid\sigma^2,x \;\sim\; N\!\left(\frac{\mu_0\sigma^2+n\bar{x}\,\sigma_0^2}{\sigma^2+n\sigma_0^2},\;\frac{\sigma^2\sigma_0^2}{\sigma^2+n\sigma_0^2}\right),$$

$$\sigma^2\mid\mu,x \;\sim\; \mathrm{IGamma}\!\left(a+\frac{n}{2},\;b+\frac{1}{2}\sum_{i=1}^{n}(x_i-\mu)^2\right).$$

---

### 5.3 Metropolis-Hastings

**Objetivo:** muestrear de $\pi(x)$ (conocida salvo constante) con propuesta $q(x,y)$ arbitraria.

#### Algoritmo 12 — Metropolis-Hastings

```
Inicializar  x_0
Para n = 1, 2, ...:
  1. Proponer   y ~ q(x_{n-1}, ·)
  2. Calcular probabilidad de aceptacion:
       alpha(x,y) = min( 1,  pi(y) * q(y,x) / (pi(x) * q(x,y)) )
  3. Generar  u ~ Uniform(0,1)
  4. Si u <= alpha(x,y):  aceptar  x_n = y
     En caso contrario:           x_n = x_{n-1}
Descartar burn-in; los restantes ~ pi(x)
```

**Correctitud** (balance detallado):

$$\pi(x)\,P_{xy} = \pi(x)\,q(x,y)\,\alpha(x,y) = \min\!\bigl(\pi(x)q(x,y),\,\pi(y)q(y,x)\bigr) = \pi(y)\,P_{yx}. \quad \blacksquare$$

**Variantes:**

| Variante | Propuesta $q(x,y)$ | Observación |
|---|---|---|
| Random Walk MH | $N(y;\,x,\,\sigma^2 I)$ | $\sigma$ requiere tuning |
| Independence Sampler | $q(y)$ independiente de $x$ | $\alpha=\min(w(y)/w(x),1)$, $w=\pi/q$ |
| Gibbs Sampling | $\pi(y_i\mid x_{-i})$ | $\alpha \equiv 1$ (siempre acepta) |

---

## Tabla Resumen de Algoritmos

| # | Algoritmo | Módulo | Propósito | Requiere |
|---|---|---|---|---|
| 1 | Inverse CDF | 1 | $X\sim F$ con $F^{-1}$ explícita | $F^{-1}$ analítica |
| 2 | Rejection Sampling | 1 | $X\sim\pi$ con propuesta envolvente | $J$, $q$, $M$ |
| 3 | Composición | 1 | Mezclas y distribuciones condicionales | Componentes $\pi_i$ |
| 4 | Cholesky Normal | 1 | $X\sim N_q(\mu,\Sigma)$ | Factorización de Cholesky |
| 5 | Newton-Raphson | 2 | Raíces de $f'=0$ / MLE | $\ell'$, $\ell''$ (o Hessiana) |
| 6 | IRLS | 2 | Ajuste de GLMs | Familia exponencial |
| 7 | Steepest Ascent | 2 | Optimización sin Hessiana | Solo gradiente $\nabla\ell$ |
| 8 | Gauss-Newton | 2 | LS no lineal | Jacobiano $h'$ |
| 9 | EM | 3 | MLE con datos latentes | $\ell_{\text{completa}}$ manejable |
| 10 | SIR | 4 | Muestra aproximada de $\pi$ | Propuesta $g$, $m\gg n$ |
| 11 | Gibbs Sampling | 5 | Posterior multivariada | Condicionales completas |
| 12 | Metropolis-Hastings | 5 | Posterior multivariada | $\pi$ salvo constante, propuesta $q$ |

---

## Referencias Bibliográficas

- Gentle, J.E. (2009). *Computational Statistics*. Springer.
- Gentle, J.E., Härdle, W.K., & Mori, Y. (2012). *Handbook of Computational Statistics*. Springer.
- Givens, G.H. & Hoeting, J.A. (2012). *Computational Statistics*. Wiley.
- Härdle, W.K., Okhrin, O., & Okhrin, Y. (2017). *Basic Elements of Computational Statistics*. Springer.
- L'Ecuyer, P. & Simard, R. (2007). TestU01: A C Library for Empirical Testing of Random Number Generators. *ACM TOMS*, 33(4). [doi:10.1145/1268776.1268777](https://doi.org/10.1145/1268776.1268777)
- Wood, S.N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). CRC Press.

---

<div align="center">

*Universidad Nacional de Ingeniería — Doctorado en Ciencias e Ingeniería Estadística*

</div>
