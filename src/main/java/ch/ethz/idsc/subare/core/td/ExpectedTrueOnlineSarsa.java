// code by fluric
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.util.EGreedyPolicy;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

public class ExpectedTrueOnlineSarsa extends TrueOnlineSarsa {
  public static TrueOnlineSarsa of( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper) {
    return new ExpectedTrueOnlineSarsa(monteCarloInterface, lambda, learningRate, featureMapper, null);
  }

  public static TrueOnlineSarsa of( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper, Tensor w) {
    return new ExpectedTrueOnlineSarsa(monteCarloInterface, lambda, learningRate, featureMapper, w);
  }

  private ExpectedTrueOnlineSarsa(MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper, Tensor w) {
    super(monteCarloInterface, lambda, learningRate, featureMapper, w);
  }

  @Override
  protected Scalar evalute(StepInterface stepInterface) {
    Tensor actions = Tensor.of( //
        monteCarloInterface.actions(stepInterface.nextState()).stream() //
            .filter(action -> learningRate.encountered(stepInterface.nextState(), action)));
    if (actions.length() == 0)
      return RealScalar.ZERO;
    // ---
    Policy policy = EGreedyPolicy.bestEquiprobable(monteCarloInterface, qsaInterface(), epsilon, stepInterface.nextState());
    return actions.stream().map(action -> policy.probability(stepInterface.nextState(), action)//
        .multiply(qsaInterface().value(stepInterface.nextState(), action)))//
        .reduce(Scalar::add).get();
  }
}
