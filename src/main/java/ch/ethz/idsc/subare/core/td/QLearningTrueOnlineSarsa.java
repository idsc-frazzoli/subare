// code by fluric
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;

/** class is instantiated via {@link SarsaType} */
/* package */ class QLearningTrueOnlineSarsa extends AbstractSharedTrueOnlineSarsa {
  QLearningTrueOnlineSarsa( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper, Tensor w) {
    super(monteCarloInterface, lambda, learningRate, featureMapper, w);
  }

  @Override // from AbstractSharedTrueOnlineSarsa
  Scalar evaluate(StepInterface stepInterface, Tensor actions) {
    return actions.stream().map(action -> qsaInterface().value(stepInterface.nextState(), action)).reduce(Max::of).get();
  }
}
