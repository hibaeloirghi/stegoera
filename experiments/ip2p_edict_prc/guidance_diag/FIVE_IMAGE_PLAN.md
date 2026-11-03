# Five new photos

The spaniel already has text scales 1.5, 3.0, 5.0, and 7.5 crossed with image scales 1.0, 1.25, and 1.5. This round does not repeat it.

Five other photos, two instructions each, at the four sender settings that separated recovery from failure on the spaniel:

| Text | Image | On the spaniel |
| ---: | ---: | --- |
| 7.5 | 1.5 | PNG recovery failed |
| 3.0 | 1.5 | PNG recovery failed |
| 7.5 | 1.0 | Detect without an exact message on one edit |
| 3.0 | 1.0 | Exact message from the PNG |

Same PRC key and the same saved noise. The sender encodes that photo and its instruction. The receivers then see only the edited PNG, the key, and the sampler settings. One of them also sees the editing instruction. Neither sees the source photo. Latents are the VAE mode and the frozen 100-step decoder inversion.

| Photo | Appearance instruction | Added-object instruction |
| --- | --- | --- |
| imagenet_dog_2 | Turn the dog into a golden retriever. | Add a red ball next to the dog. |
| imagenet_cat | Turn the cat into a black cat. | Add a red ball next to the cat. |
| church | Turn the concrete church into a red brick church. | Add a red bus in front of the church. |
| fountain | Turn the turquoise fountain into a white marble fountain. | Add a red ball in the fountain basin. |
| imagenet_lake | Turn the lake into a frozen lake. | Add a small red boat on the lake. |
