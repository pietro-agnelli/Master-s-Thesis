# Commento codice UR10 con CBF

Questo documento esplicita **le equazioni utilizzate** nel codice fornito, con ragionamenti e spiegazioni su come e perché sono state usate, senza soffermarsi sulla struttura del programma ma solo sulla **logica di controllo, sulle scelte matematiche e sulle Control Barrier Functions (CBF)**.

---

## 1. **Equazioni di sicurezza ISO/TS 15066**

### **a) Velocità massima consentita in funzione della distanza**

Nel controllo di sicurezza robotica collaborativa, la ISO/TS 15066 impone che la velocità massima del robot dipenda dalla distanza da una persona. Si vuole garantire che, in caso di arresto di emergenza, la distanza minima di sicurezza non venga violata.

**Equazione:**

$$
v_{max}(S) = \max \left(0, \sqrt{v_h^2 + (a_s T_r)^2 - 2 a_s (C - S)} - a_s T_r - v_h \right)
$$

dove:

- \(S\): distanza attuale robot-umano/ostacolo
- \(C\): distanza minima di sicurezza (fissata)
- \(T_r\): tempo di reazione del controller
- \(a_s\): decelerazione massima del robot
- \(v_h\): velocità massima di avvicinamento dell’umano

**Origine:**
Questa equazione nasce dalla conservazione dello spazio necessario perché robot e umano si fermino in sicurezza:

$$
S \geq C + (v_{max} + v_h)T_r + \frac{v_{max}^2}{2a_s} + v_h \frac{v_{max}}{a_s}
$$

Risolvendo per \(v_{max}\) (vedi risposta precedente), si ottiene la forma quadratica sopra implementata in `vmax`.

---

### **b) Distanza minima consentita in funzione della velocità relativa**

La funzione `s_min` calcola invece **la distanza minima richiesta** per garantire sicurezza, dato che il robot si muove a una certa velocità rispetto all’ostacolo/umano.

**Equazione:**
I diversi rami rappresentano casi di moto relativo diverso

$$
  v \leq 0 :  s_{min} = C + v_h T_r - v T_r + v_h \left(\frac{ -v }{ a_s } \right) + \frac{1}{2} \frac{v^2}{a_s}
$$

$$
0< v\leq v_h : s_{min} = C + v_h T_r - v T_r - \frac{1}{2}(v + v_h)\frac{v_h - v}{a_s} + v_h \frac{v_h - v}{a_s}
$$

$$
v > v_h : s_{min} = C - v T_r + v_h T_r
$$

**Origine:**
Sono tutte derivate dall’equazione energetica della frenata e dall’analisi dei tempi di reazione e di arresto di entrambi (vedi deduzioni precedenti).

---

### **c) Derivata di s\_min rispetto a v\_rel**

Per la CBF serve anche la derivata di \(s_{min}\) rispetto alla velocità relativa:

$$
\frac{\partial s_{min}}{\partial v} =
\begin{cases}
    -T_r + \frac{v - v_h}{a_s} & \text{se } v < v_h \\
    -T_r & \text{se } v \geq v_h
\end{cases}
$$

Nel codice la funzione `dh_dvrel` restituisce \(-\frac{\partial s_{min}}{\partial v}\), che serve per la derivata totale di h (vedi sotto).

---

## 2. **Dinamica della distanza: derivata rispetto a stato e controllo**

La funzione `range_state_derivative` sviluppa le derivate di distanza e della sua evoluzione:

$$
d(x) = \| r \| \qquad (r = posizione\ robot - posizione\ ostacolo)
$$

- **Derivata prima della distanza:**

$$
\dot{d} = \frac{ r \cdot v }{ \| r \| }
$$

- **Derivata seconda:**

$$
\ddot{d} = \frac{v \cdot v}{\| r \|} - \frac{ ( r \cdot v )^2 }{ \| r \|^3 }
$$

Queste servono a esprimere la dinamica della distanza nella formulazione della CBF.

---

## 3. **Definizione e uso della CBF**

### **a) Funzione di barriera**

Nel codice la **funzione di barriera** è:

$$
h(x) = d(x) - s_{min}(v_{rel})
$$

dove:

- \(d(x)\): distanza attuale robot-ostacolo
- \(s_{min}(v_{rel})\): distanza minima ammessa per la velocità relativa attuale

**Significato:**

- \(h(x) > 0\) → il sistema è sicuro (rispetta il margine di sicurezza)
- \(h(x) < 0\) → rischio di collisione


### **b) Vincolo sulla derivata di h (CBF condition)**

La CBF richiede che la derivata di \(h\) sia sempre sufficientemente positiva:

$$
\dot{h}(x) + \gamma h(x) \geq 0
$$

dove \(\gamma\) è il "CBF gain" (scelta di tuning: più è grande, più la barriera è reattiva).

**Sviluppando la derivata di h:**

$$
\dot{h} = \frac{\partial d}{\partial x}\dot{x} - \frac{\partial s_{min}}{\partial v_{rel}} \cdot \dot{v}_{rel}
$$

Nel controllo, questa disuguaglianza viene scritta in funzione delle variabili di controllo (accelerazioni dei giunti) e della dinamica del robot, tramite Jacobiani e derivate.

Il codice calcola i termini di Lie:

- **Lie\_f\_h:** parte naturale (senza input)
- **Lie\_g\_h:** parte che dipende direttamente dal controllo

e costruisce la CBF come vincolo lineare nella QP:

$$
(Lie\_g\_h \cdot J_{lin}) \cdot \ddot{q} \geq -Lie\_g\_h \cdot \dot{J}_{lin} \cdot \dot{q} - Lie\_f\_h - \gamma h
$$

### **c) Definizione CBF nello script**
```python
for i, obs_pos in enumerate(obstacle_positions):
    # update obstacle motion
    w1=2 * np.pi/2
    w2=2 * np.pi/2.1
    obs_pos[0] = 0.8 - 0.25 * np.sin(w1* t)
    obs_pos[1] = 0.4 + 0.1 * np.sin(w2 * t)
    v_obs=np.array([0]*3)
    v_obs[0] = -0.25 * np.cos(w1* t)*w1
    v_obs[1] = 0.1 * np.cos(w2 * t)*w2

    # Calcolo distanza e velocità relativa tra robot e ostacolo
    r = translation_bt - obs_pos
    distance = np.linalg.norm(r)
    u_hr = r / distance
    v_h = np.dot(u_hr,v_obs)
    v_rel = np.dot(vel_lineare, u_hr)

    # Calcolo distanza minima di sicurezza
    smin = s_min(v=v_rel, v_h=v_h)
    h = distance - smin  # funzione di barriera CBF

    # Derivate necessarie per la CBF
    f, g = range_state_derivative(r, vel_lineare)#dinamica 
    derivative_h_on_distance = 1.0
    derivative_h_on_velocity = dh_dvrel(v=v_rel, v_h=v_h)
    partial_h_on_x = np.array([derivative_h_on_distance, derivative_h_on_velocity]).reshape(1, -1)
    Lie_f_h = partial_h_on_x @ f  # termine naturale
    Lie_g_h = partial_h_on_x @ g  # termine di controllo

    # Vincolo lineare (una riga della QP)
    constraint_matrix = np.append(constraint_matrix, Lie_g_h @ Jlin, axis=0)
    constraint_vector = np.append(
        constraint_vector,
        -Lie_g_h @ dJlin @ dq - Lie_f_h - gamma * h,
        axis=0,
    )
```

---

## 4. **Controllo e ottimizzazione (QP)**

L’obiettivo è trovare l’accelerazione giunto $\ddot{q}$ che minimizza la differenza tra accelerazione desiderata (controllo PD in spazio cartesiano):

$$
\min_{\ddot{q}} \| J \ddot{q} + \dot{J}\dot{q} - \ddot{x}_{des} \|^2
$$

soggetto ai vincoli CBF (se abilitati):

$$
(Lie\_g\_h \cdot J_{lin}) \cdot \ddot{q} \geq ...
$$

Relazione cinematica (TCP): $\ddot x = J\,\ddot q + \dot J\,\dot q.$ 

Errore da minimizzare: $e(\ddot q) = J\,\ddot q + \dot J\,\dot q - \ddot x_{des}.$

Least‑squares: $\min_{\ddot q} \tfrac12 \|e(\ddot q)\|^2.$ 

Espansione ⇒ forma quadratica **standard**:

$$
\tfrac12 \ddot q^{\!\top} J^{\!\top}J \ddot q - \bigl(J^{\!\top}(\ddot x_{des}-\dot J\dot q)\bigr)^{\!\top}\ddot q.
$$

Quindi $\boxed{P = J^{\!\top}J},\quad \boxed{b = J^{\!\top}(\ddot x_{des}-\dot J\dot q)}.$

Codice:
``` python
P = J.T @ J
        b = (J.T @ (dtwist_tool - dJ @ dq)).flatten()
        constraint_vector = constraint_vector.flatten()

        if CBF:
            try:
                ddq, *_ = quadprog.solve_qp(
                    P,
                    b,
                    constraint_matrix.T,
                    constraint_vector,
                    0,
                )
            except ValueError as err:
                if "constraints are inconsistent" in str(err):
                    print("QP infeasible – applying fallback damping.")
                    ddq = -10.0 * dq
                else:
                    raise
        else:
            ddq = damped_pinv_svd(J) @ (dtwist_tool - dJ @ dq)
```
`quadprog` risolve la disuguaglianza come $Gᵀ x ≥ h$.

---

## 5. **Controllo PD nello spazio cartesiano**

La parte di tracking calcola accelerazioni desiderate per posizione e orientamento:

- **Posizione:**

$$
\ddot{x}_{lin,des} = K_p (x_{goal} - x_{curr}) + K_d (\dot{x}_{goal} - \dot{x}_{curr})
$$

- **Orientamento:** (in logaritmo di rotazione)

$$
\ddot{\theta}_{des} = K_{p,rot} (\text{error\_rot}) + K_{d,rot} (\omega_{goal} - \omega_{curr})
$$

Questi vengono concatenati in `dtwist_tool`.

---

## 6. **Integrazione delle accelerazioni**

Le accelerazioni trovate vengono integrate per aggiornare posizione e velocità dei giunti:

$$
q_{k+1} = q_k + \dot{q}_k T_c + \frac{1}{2} \ddot{q}_k T_c^2
$$

$$
\dot{q}_{k+1} = \dot{q}_k + \ddot{q}_k T_c
$$

dove \(T_c\) è il periodo di controllo (2ms qui).

---
## Caso CBF OFF 
Quando non vogliamo imporre il vincolo dato dalla CBF allora il problema si riduce a mantere la $\ddot{x} = J\dot{q} + \dot{J}q$ 
il più vicina possibile alla $\ddot{x}_{des}$
