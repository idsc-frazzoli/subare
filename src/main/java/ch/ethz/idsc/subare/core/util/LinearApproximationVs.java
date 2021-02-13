// code by jph
package ch.ethz.idsc.subare.core.util;

import java.io.Serializable;

import ch.ethz.idsc.subare.core.VsInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.api.TensorUnaryOperator;

public class LinearApproximationVs implements VsInterface, Serializable {
  /** @param represent
   * @param weight initial choice of weights
   * @return */
  public static VsInterface create(TensorUnaryOperator represent, Tensor weight) {
    return new LinearApproximationVs(represent, weight);
  }

  /***************************************************/
  private final TensorUnaryOperator represent;
  private Tensor weight;

  private LinearApproximationVs(TensorUnaryOperator represent, Tensor weight) {
    this.represent = represent;
    this.weight = weight;
  }

  @Override // from VsInterface
  public Scalar value(Tensor state) {
    Tensor gradient = represent.apply(state);
    return (Scalar) gradient.dot(weight);
  }

  @Override // from VsInterface
  public void increment(Tensor state, Scalar delta) {
    Tensor gradient = represent.apply(state);
    weight = weight.add(gradient.multiply(delta));
  }

  @Override // from VsInterface
  public VsInterface copy() {
    return new LinearApproximationVs(represent, weight.copy());
  }

  @Override // from VsInterface
  public VsInterface discounted(Scalar gamma) {
    return new LinearApproximationVs(represent, weight.multiply(gamma));
  }
}
