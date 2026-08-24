#if defined(__aarch64__)

    .text
    .p2align 4

/* Transform one signed O4I4 block into four centered weight columns. */
    .macro CAMPP_QCONV_LOAD_COLUMNS weight_pointer, zero_offset
    ldr q28, [\weight_pointer], #16
    sshll v29.8h, v28.8b, #0
    sshll2 v30.8h, v28.16b, #0
    ext v31.16b, v29.16b, v29.16b, #8
    trn1 v4.4h, v29.4h, v31.4h
    trn2 v5.4h, v29.4h, v31.4h
    ext v31.16b, v30.16b, v30.16b, #8
    trn1 v6.4h, v30.4h, v31.4h
    trn2 v7.4h, v30.4h, v31.4h
    /*
     * v4/v5 carry input lanes {0,2}/{1,3} of outputs 0-1 and v6/v7 the same
     * lanes of outputs 2-3, so pairing them yields columns in lane order
     * 0, 2, 1, 3. CAMPP_QCONV_ACCUMULATE feeds v0..v3 to input lanes 0..3, so
     * lanes 1 and 2 must land in v1 and v2 respectively.
     */
    trn1 v0.2s, v4.2s, v6.2s
    trn2 v2.2s, v4.2s, v6.2s
    trn1 v1.2s, v5.2s, v7.2s
    trn2 v3.2s, v5.2s, v7.2s
    ldr q31, [x4, #\zero_offset]
    xtn v31.4h, v31.4s
    sub v0.4h, v0.4h, v31.4h
    sub v1.4h, v1.4h, v31.4h
    sub v2.4h, v2.4h, v31.4h
    sub v3.4h, v3.4h, v31.4h
    .endm

    .macro CAMPP_QCONV_ACCUMULATE accumulator, input
    smlal \accumulator\().4s, v0.4h, \input\().h[0]
    smlal \accumulator\().4s, v1.4h, \input\().h[1]
    smlal \accumulator\().4s, v2.4h, \input\().h[2]
    smlal \accumulator\().4s, v3.4h, \input\().h[3]
    .endm

    .global campp_qconv_mac_4x8_aarch64_raw
    .type campp_qconv_mac_4x8_aarch64_raw, %function
campp_qconv_mac_4x8_aarch64_raw:
    ldr x1, [x0, #0]
    ldr x2, [x0, #8]
    ldr x3, [x0, #16]
    ldr x4, [x0, #24]
    ldr x5, [x0, #32]
    ldr w6, [x0, #40]
    ldr w7, [x0, #44]
    ldr w8, [x0, #48]
    lsr w7, w7, #2

    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0

.Lcampp_qconv_4x8_kernel:
    ldp x10, x11, [x1]
    ldp x12, x13, [x1, #16]
    mov w14, #0
    mov w15, w7

.Lcampp_qconv_4x8_input_block:
    ldr s24, [x10, x14]
    ldr s25, [x11, x14]
    ldr s26, [x12, x14]
    ldr s27, [x13, x14]
    uxtl v24.8h, v24.8b
    uxtl v25.8h, v25.8b
    uxtl v26.8h, v26.8b
    uxtl v27.8h, v27.8b
    dup v31.4h, w8
    sub v24.4h, v24.4h, v31.4h
    sub v25.4h, v25.4h, v31.4h
    sub v26.4h, v26.4h, v31.4h
    sub v27.4h, v27.4h, v31.4h

    CAMPP_QCONV_LOAD_COLUMNS x2, 0
    /* SMLAL-by-element encodes its lane source in v0-v15 only. */
    mov v4.8b, v24.8b
    mov v5.8b, v25.8b
    mov v6.8b, v26.8b
    mov v7.8b, v27.8b
    CAMPP_QCONV_ACCUMULATE v16, v4
    CAMPP_QCONV_ACCUMULATE v17, v5
    CAMPP_QCONV_ACCUMULATE v18, v6
    CAMPP_QCONV_ACCUMULATE v19, v7

    CAMPP_QCONV_LOAD_COLUMNS x3, 16
    mov v4.8b, v24.8b
    mov v5.8b, v25.8b
    mov v6.8b, v26.8b
    mov v7.8b, v27.8b
    CAMPP_QCONV_ACCUMULATE v20, v4
    CAMPP_QCONV_ACCUMULATE v21, v5
    CAMPP_QCONV_ACCUMULATE v22, v6
    CAMPP_QCONV_ACCUMULATE v23, v7

    add w14, w14, #4
    subs w15, w15, #1
    b.ne .Lcampp_qconv_4x8_input_block
    add x1, x1, #64
    subs w6, w6, #1
    b.ne .Lcampp_qconv_4x8_kernel

    stp q16, q20, [x5]
    stp q17, q21, [x5, #32]
    stp q18, q22, [x5, #64]
    stp q19, q23, [x5, #96]
    mov w0, #0
    ret
    .size campp_qconv_mac_4x8_aarch64_raw, .-campp_qconv_mac_4x8_aarch64_raw

#endif

    .section .note.GNU-stack,"",%progbits
