// code by jph
package ch.ethz.idsc.subare.core.adapter;

import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.api.TensorUnaryOperator;
import ch.ethz.idsc.tensor.sca.Clip;
import ch.ethz.idsc.tensor.sca.Cos;

/** 9.5.2 Fourier Basis p.171
 * 
 * univariate basis functions on the unit interval */
public class CosineBasis implements TensorUnaryOperator {
  /** @param order number of basis functions
   * @param clip */
  public static TensorUnaryOperator create(int order, Clip clip) {
    return new CosineBasis(order, clip);
  }

  /***************************************************/
  private final int order;
  private final Clip clip;

  private CosineBasis(int order, Clip clip) {
    this.order = order;
    this.clip = clip;
  }

  @Override // from UnaryOperator
  public Tensor apply(Tensor tensor) {
    Scalar param = clip.requireInside((Scalar) tensor);
    Scalar value = clip.rescale(param);
    return Tensors.vector(i -> Cos.FUNCTION.apply(DoubleScalar.of(i * Math.PI).multiply(value)), order);
  }
}
