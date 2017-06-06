// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** suggested base class for implementations of sarsa and expected sarsa */
public abstract class Sarsa implements StepDigest {
  final DiscreteModel discreteModel;
  final QsaInterface qsa;
  private final Scalar gamma;
  private final Scalar alpha;

  /** @param discreteModel
   * @param qsa
   * @param alpha */
  public Sarsa(DiscreteModel discreteModel, QsaInterface qsa, Scalar alpha) {
    this.discreteModel = discreteModel;
    this.qsa = qsa;
    this.gamma = discreteModel.gamma();
    this.alpha = alpha;
  }

  @Override
  public final void digest(StepInterface stepInterface) {
    Tensor state0 = stepInterface.prevState();
    Tensor action0 = stepInterface.action();
    Scalar reward = stepInterface.reward();
    Tensor state1 = stepInterface.nextState();
    Scalar value0 = qsa.value(state0, action0);
    // ---
    Scalar value1 = evaluate(state1); // <- call implementation
    // ---
    Scalar delta = reward.add(gamma.multiply(value1)).subtract(value0).multiply(alpha);
    qsa.assign(state0, action0, value0.add(delta));
  }

  /** @param state1
   * @return value of state1 */
  protected abstract Scalar evaluate(Tensor state1);
}
