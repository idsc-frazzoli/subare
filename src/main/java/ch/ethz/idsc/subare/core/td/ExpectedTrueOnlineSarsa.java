// code by fluric
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

/** class is instantiated via {@link SarsaType} */
/* package */ class ExpectedTrueOnlineSarsa extends TrueOnlineSarsa {
  ExpectedTrueOnlineSarsa( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper, Tensor w) {
    super(monteCarloInterface, lambda, learningRate, featureMapper, w);
  }

  @Override // from AbstractSharedTrueOnlineSarsa
  Scalar evaluate(Tensor state, Tensor actions) {
    QsaInterface qsaInterface = qsaInterface();
    Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, qsaInterface, epsilon, state);
    return actions.stream() //
        .map(action -> policy.probability(state, action).multiply(qsaInterface.value(state, action))) //
        .reduce(Scalar::add) //
        .get();
  }
}
