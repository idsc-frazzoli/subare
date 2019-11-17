// code by jph
package ch.ethz.idsc.subare.demo.prison;

import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** Julian's idea: Prisoners' Dilemma */
/* package */ enum Arena {
  ;
  /** rewards average at 2 */
  static final Tensor R0 = Tensors.matrixInt(new int[][] { //
      { -1, 2 }, //
      { -2, 1 } }).multiply(RationalScalar.HALF);
}
