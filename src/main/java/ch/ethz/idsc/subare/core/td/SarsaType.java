// code by fluric, jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StateActionCounter;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.FeatureWeight;
import ch.ethz.idsc.subare.core.util.LearningRate;
import ch.ethz.idsc.subare.core.util.PolicyBase;
import ch.ethz.idsc.tensor.Scalar;

public enum SarsaType {
  ORIGINAL() {
    @Override
    SarsaEvaluation sarsaEvaluation(DiscreteModel discreteModel) {
      return new OriginalSarsaEvaluation(discreteModel);
    }
  }, //
  EXPECTED() {
    @Override
    SarsaEvaluation sarsaEvaluation(DiscreteModel discreteModel) {
      return new ExpectedSarsaEvaluation(discreteModel);
    }
  }, //
  QLEARNING() {
    @Override
    SarsaEvaluation sarsaEvaluation(DiscreteModel discreteModel) {
      return new QLearningSarsaEvaluation(discreteModel);
    }
  }, //
  ;
  // ---
  abstract SarsaEvaluation sarsaEvaluation(DiscreteModel discreteModel);

  public Sarsa sarsa( //
      DiscreteModel discreteModel, //
      LearningRate learningRate, //
      QsaInterface qsa, //
      StateActionCounter sac, //
      PolicyBase policy) {
    return new Sarsa(sarsaEvaluation(discreteModel), discreteModel, learningRate, qsa, sac, policy);
  }

  public DoubleSarsa doubleSarsa( //
      DiscreteModel discreteModel, //
      LearningRate learningRate, //
      QsaInterface qsa1, QsaInterface qsa2, //
      StateActionCounter sac1, StateActionCounter sac2, //
      PolicyBase policy1, PolicyBase policy2) {
    return new DoubleSarsa(sarsaEvaluation(discreteModel), discreteModel, learningRate, qsa1, qsa2, sac1, sac2, policy1, policy2);
  }

  /** @param monteCarloInterface
   * @param lambda in [0, 1] Figure 12.14 in the book suggests that lambda in [0.8, 0.9] tends to be a good choice */
  public TrueOnlineSarsa trueOnline( //
      MonteCarloInterface monteCarloInterface, //
      Scalar lambda, //
      FeatureMapper featureMapper, //
      LearningRate learningRate, //
      FeatureWeight w, //
      StateActionCounter sac, //
      PolicyBase policy) {
    return new TrueOnlineSarsa(monteCarloInterface, sarsaEvaluation(monteCarloInterface), lambda, featureMapper, learningRate, w, sac, policy);
  }

  public DoubleTrueOnlineSarsa doubleTrueOnline( //
      MonteCarloInterface monteCarloInterface, //
      Scalar lambda, //
      FeatureMapper featureMapper, //
      LearningRate learningRate, //
      FeatureWeight w1, FeatureWeight w2, //
      StateActionCounter sac1, StateActionCounter sac2, //
      PolicyBase policy1, PolicyBase policy2) {
    return new DoubleTrueOnlineSarsa( //
        monteCarloInterface, //
        sarsaEvaluation(monteCarloInterface), //
        lambda, featureMapper, learningRate, w1, w2, sac1, sac2, policy1, policy2);
  }
}
