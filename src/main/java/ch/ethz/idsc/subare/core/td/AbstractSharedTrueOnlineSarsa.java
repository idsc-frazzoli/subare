// code by fluric
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

abstract class AbstractSharedTrueOnlineSarsa extends TrueOnlineSarsa {
  AbstractSharedTrueOnlineSarsa( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper, Tensor w) {
    super(monteCarloInterface, lambda, learningRate, featureMapper, w);
  }

  @Override // from TrueOnlineSarsa
  protected final Scalar evaluate(StepInterface stepInterface) {
    Tensor nextState = stepInterface.nextState();
    Tensor actions = Tensor.of( //
        monteCarloInterface.actions(nextState).stream() //
            .filter(action -> learningRate.encountered(nextState, action)));
    return Tensors.isEmpty(actions) //
        ? RealScalar.ZERO
        : evaluate(stepInterface, actions);
  }

  abstract Scalar evaluate(StepInterface stepInterface, Tensor actions);
}
