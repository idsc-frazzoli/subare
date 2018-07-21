// code by fluric
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.PolicyWrap;
import ch.ethz.idsc.subare.core.util.StateAction;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

class OriginalTrueOnlineSarsa extends TrueOnlineSarsa {
  static TrueOnlineSarsa of( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper) {
    return new OriginalTrueOnlineSarsa(monteCarloInterface, lambda, learningRate, featureMapper, null);
  }

  static TrueOnlineSarsa of( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper, Tensor w) {
    return new OriginalTrueOnlineSarsa(monteCarloInterface, lambda, learningRate, featureMapper, w);
  }

  // ---
  OriginalTrueOnlineSarsa(MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper, Tensor w) {
    super(monteCarloInterface, lambda, learningRate, featureMapper, w);
  }

  @Override // from TrueOnlineSarsa
  protected Scalar evaluate(StepInterface stepInterface) {
    Tensor nextState = stepInterface.nextState();
    Tensor actions = Tensor.of( //
        monteCarloInterface.actions(nextState).stream() //
            .filter(action -> learningRate.encountered(nextState, action)));
    if (Tensors.isEmpty(actions))
      return RealScalar.ZERO;
    // ---
    Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, qsaInterface(), epsilon, nextState);
    Tensor nextAction = new PolicyWrap(policy).next(nextState, actions);
    Tensor nextX = featureMapper.getFeature(StateAction.key(nextState, nextAction));
    return w.dot(nextX).Get();
  }
}
