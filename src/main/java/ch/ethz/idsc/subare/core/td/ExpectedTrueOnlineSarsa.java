// code by fluric
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** class is instantiated via {@link SarsaType} */
/* package */ class ExpectedTrueOnlineSarsa extends AbstractSharedTrueOnlineSarsa {
  ExpectedTrueOnlineSarsa( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper, Tensor w) {
    super(monteCarloInterface, lambda, learningRate, featureMapper, w);
  }

  @Override // from AbstractSharedTrueOnlineSarsa
  Scalar evaluate(StepInterface stepInterface, Tensor actions) {
    Tensor nextState = stepInterface.nextState();
    QsaInterface qsaInterface = qsaInterface();
    Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, qsaInterface, epsilon, nextState);
    return actions.stream() //
        .map(action -> policy.probability(nextState, action).multiply(qsaInterface.value(nextState, action))) //
        .reduce(Scalar::add) //
        .get();
  }
}
