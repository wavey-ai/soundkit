#include <stdio.h>

#include "entcode.h"
#include "entdec.h"
#include "entenc.h"
#include "vq.h"

int main(void) {
    unsigned char packet[64] = {0};
    celt_norm input[8] = {0.12f, -0.27f, 0.43f, -0.91f, 0.35f, 0.08f, -0.52f, 0.61f};
    celt_norm decoded[8] = {0};
    ec_enc encoder;
    ec_dec decoder;

    ec_enc_init(&encoder, packet, sizeof(packet));
    unsigned encode_mask = cubic_quant(input, 8, 4, 1, &encoder, 1.0f, 1);
    int tell = ec_tell(&encoder);
    unsigned bytes = (unsigned)(tell + 7) / 8;
    ec_enc_shrink(&encoder, bytes);
    ec_enc_done(&encoder);

    printf("mask=%u tell=%d bytes=%u packet=", encode_mask, tell, bytes);
    for (unsigned i = 0; i < bytes; i++) printf("%02x", packet[i]);
    printf("\nresynth=");
    for (int i = 0; i < 8; i++) printf("%s%.9g", i ? "," : "", input[i]);

    ec_dec_init(&decoder, packet, bytes);
    unsigned decode_mask = cubic_unquant(decoded, 8, 4, 1, &decoder, 1.0f);
    printf("\ndecode_mask=%u decoded=", decode_mask);
    for (int i = 0; i < 8; i++) printf("%s%.9g", i ? "," : "", decoded[i]);
    printf("\n");
    return 0;
}
