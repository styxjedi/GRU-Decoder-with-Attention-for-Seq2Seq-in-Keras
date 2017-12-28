# GRU-Decoder-with-Attention-for-Seq2Seq-in-Keras
A simple GRU decoder cell for seq2seq model in Keras.

## How to use:

> from keras.layers import RNN
> from AttnDecoder import AttnDecoderCell
> cell = AttnDecoderCell(output_dims)
> decoder_layer = RNN(cell, return_sequences=True)
> decoder_outputs = decoder_layer(input_tensor, initial_state=init_state, constants=[attn_sequence])

where *init_state* is the first hidden state of decoder, and *attn_sequence* is the sequence that needs attention mechanism.
