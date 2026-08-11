Il **primo modello** (`ROIClassifier`) è una rete neurale progettata per la **classificazione di regioni di interesse (ROI)** all'interno di un'immagine — ad esempio per identificare interazioni uomo-oggetto (*Human-Object Interaction* o HOI).

Ecco come funziona passo dopo passo:

1. **Gestione dell'input speciale (5 canali):**
Invece di un'immagine standard a 3 canali (RGB), accetta un input a **5 canali** (ad esempio: i 3 canali RGB dell'immagine più 2 canali di maschere o informazioni sul contesto spaziale della ROI).
2. **Riduzione dei canali ($5 \to 3$):**
Usa due strati di convoluzione $1 \times 1$ (`self.first` e `self.pre_conv`) per comprimere i 5 canali iniziali e riportarli a **3 canali**, rendendo il dato compatibile con i modelli preaddestrati standard.
3. **Estrae le caratteristiche con una ResNet-18:**
Passa la mappa a 3 canali attraverso la spina dorsale (**backbone**) della **ResNet-18** preaddestrata. Rimuovendo l'ultimo strato di classificazione (`self.backbone.fc = nn.Identity()`), la rete trasforma l'immagine in un vettore compatto di **512 caratteristiche visive**.
4. **Classificazione finale:**
Usa uno strato lineare (`self.fc`) per convertire le 512 caratteristiche nelle classi di destinazione (`num_hoi_classes`) e applica la funzione di attivazione **Sigmoide** per restituire una probabilità per ogni classe.

---

In sintesi: il primo modello prende **dati complessi con contesto (5 canali)**, li adatta e usa la classica architettura convoluzionale **ResNet-18** per estrarne il significato ed emettere una classificazione.

This implementation replaces the convolutional ResNet-18 backbone with a Vision Transformer (ViT) architecture from Hugging Face, while keeping the 5-to-3 channel reduction preprocessing identical.

| Feature | Previous ResNet-18 Version | New ViT-Base Version |
| --- | --- | --- |
| **Backbone Architecture** | Convolutional Neural Network (`torchvision.models.resnet18`) | Vision Transformer (`transformers.ViTModel`) |
| **Feature Dimension** | 512 features from global pooling | 768 hidden dimensions |
| **Pooling / Representation** | Global average pooling / Identity layer | `[CLS]` token representation (`outputs.last_hidden_state[:, 0]`) |
| **Required Libraries** | `torchvision` | `transformers` (Hugging Face ecosystem) |
| **Channel Reduction** | $5 \to 4 \to 3$ via consecutive $1 \times 1$ convs | Identical $5 \to 4 \to 3$ via consecutive $1 \times 1$ convs |

### Key Trade-offs to Consider

* **Spatial Processing:** ResNet relies on local receptive fields and translation equivariance via convolutions. ViT splits the image into $16 \times 16$ patches and models global context immediately through self-attention layers.
* **Input Resolution Constraints:** ViT models (like `google/vit-base-patch16-224`) expect fixed-size inputs ($224 \times 224$ pixels) matching their patch configurations, whereas fully convolutional networks like ResNet-18 are naturally more flexible with varying input spatial dimensions.