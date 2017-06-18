//// code by jph
// package ch.ethz.idsc.subare.util;
//
// import ch.ethz.idsc.tensor.RealScalar;
// import ch.ethz.idsc.tensor.Scalar;
// import ch.ethz.idsc.tensor.Tensor;
// import ch.ethz.idsc.tensor.alg.Multinomial;
// import ch.ethz.idsc.tensor.red.Total;
//
// public enum FastHorner {
// ;
// public static Scalar of(Tensor coeffs, Scalar scalar) {
// return scalar.equals(RealScalar.ONE) ? //
// Total.of(coeffs).Get() : Multinomial.horner(coeffs, scalar);
// }
// }
