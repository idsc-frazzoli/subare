// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public abstract class AbstractSarsa extends AbstractTemporalDifference {
  final DiscreteModel discreteModel;
  final QsaInterface qsa;
  private final Scalar gamma;
  private final Scalar alpha;

  public AbstractSarsa( //
      EpisodeSupplier episodeSupplier, PolicyInterface policyInterface, //
      DiscreteModel discreteModel, //
      QsaInterface qsa, Scalar gamma, Scalar alpha) {
    super(episodeSupplier, policyInterface);
    this.discreteModel = discreteModel;
    this.qsa = qsa;
    this.gamma = gamma;
    this.alpha = alpha;
  }

  @Override
  public final void digest(StepInterface stepInterface) {
    // ---
    Tensor state0 = stepInterface.prevState();
    Tensor action0 = stepInterface.action();
    Scalar reward = stepInterface.reward();
    Scalar value0 = qsa.value(state0, action0);
    Scalar value1 = evaluate(stepInterface.nextState()); // <- call implementation
    Scalar delta = reward.add(value1.multiply(gamma)).subtract(value0).multiply(alpha);
    qsa.assign(state0, action0, value0.add(delta));
  }

  protected abstract Scalar evaluate(Tensor state1);
}