// code by jph
package ch.ethz.idsc.subare.ch02;

import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** the random agent picks any action equally likely
 * the policy is a constant vector of pi(a)=1/n */
public class RandomAgent extends Agent {
  final int n;

  public RandomAgent(int n) {
    this.n = n;
  }

  @Override
  public int takeAction() {
    notifyAboutRandomizedDecision();
    return random.nextInt(n);
  }

  @Override
  protected void protected_feedback(int a, Scalar value) {
    // ---
  }

  @Override
  protected Tensor protected_QValues() {
    return Tensors.vector(i -> RationalScalar.of(1, n), n);
  }

  @Override
  public String getDescription() {
    return "";
  }
}
