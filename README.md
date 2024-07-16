Abstract

L’Industria 4.0 rappresenta una rivoluzione nell’ambito industriale, caratteriz-
zata dall’uso di tecnologie digitali avanzate nei processi produttivi. L’obiettivo

dell’Industria 4.0 `e creare fabbriche intelligenti e connesse, in grado di operare in
modo autonomo, comunicando tra di loro e adattandosi dinamicamente alle esigenze
del mercato, aumentando cos`ı l’efficienza e la competitivit`a dell’industria.

L’Intelligenza Artificiale (AI) e l’Intelligenza Artificiale spiegabile (XAI), svol-
gono un ruolo cruciale in questo contesto: la prima permette di ottimizzare i pro-
cessi produttivi e decisionali attraverso l’impiego di algoritmi avanzati, mentre la

seconda consente agli operatori del settore di comprendere il funzionamento che si
cela dietro a questi sistemi intelligenti, facilitando l’adozione e l’integrazione dell’IA

nell’Industria 4.0 e promuovendo una collaborazione pi`u efficace tra umani e mac-
chine.

Il lavoro di questa tesi si concentra in particolare sullo studio del rilevamento
delle anomalie (anomaly detection): una tecnica di analisi di comportamenti anomali

all’interno un processo industriale, al fine di prevedere malfunzionamenti nei macchi-
nari industriali e garantire un’alta efficienza dell’intero ciclo produttivo.

L’obiettivo `e stato analizzare un insieme di dati elaborati provenienti dal set-
tore delle lavanderie industriali, cercando di rilevare quando un gruppo di dati

nello specifico potesse essere definito “anomalo” rispetto a tutti gli altri valori. Per

fare questo lavoro sono stati usati inizialmente due algoritmi di AI: MLP (Mul-
tilayer Perceptron), basato su rete neurale artificiale, ed Isolation Forest, basato

invece su alberi decisionali e sviluppato in particolare per studiare il problema
riguardante il rilevamento di anomalie. In seguito, tramite alcune tecniche di XAI,
l’analisi si `e concentrata sul capire quale delle singole grandezze del dataset avessero
un’importanza relativa maggiore nel determinare un’anomalia. Nello specifico,
l’impiego di alcune librerie durante questa analisi ha reso necessaria l’estensione

di alcune funzionalit`a del modello Isolation Forest. Questo ha portato ad un ul-
teriore sviluppo del lavoro, con l’obiettivo di trovare una soluzione che potesse

integrarsi con il modello preesistente, dovendo poi testarla valutando il suo fun-
zionamento anche in maniera pi`u sistematica. Grazie all’utilizzo di tecniche di XAI

`e stato possibile ottenere una maggiore comprensione dei dati e fornire anche al-
cune spiegazioni sui due modelli utilizzati, rapportando l’efficacia di ognuno di essi

in funzione del problema del rilevamento delle anomalie e dei dati analizzati.
