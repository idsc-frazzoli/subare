// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.subare.core.util.FeatureWeight;
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

  // TODO JAN rename function
  public final Sarsa supply(DiscreteModel discreteModel, LearningRate learningRate, QsaInterface qsa) {
    return new Sarsa(sarsaEvaluation(discreteModel), discreteModel, learningRate, qsa);
  }

  public final DoubleSarsa doubleSarsa( //
      DiscreteModel discreteModel, LearningRate learningRate1, LearningRate learningRate2, QsaInterface qsa1, QsaInterface qsa2) {
    return new DoubleSarsa(sarsaEvaluation(discreteModel), discreteModel, learningRate1, learningRate2, qsa1, qsa2);
  }

  /** @param monteCarloInterface
   * @param lambda in [0, 1] Figure 12.14 in the book suggests that lambda in [0.8, 0.9] tends to be a good choice
   * @param learningRate
   * @param featureMapper
   * @param w */
  public final TrueOnlineSarsa trueOnline( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, FeatureMapper featureMapper, LearningRate learningRate, FeatureWeight w) {
    return new TrueOnlineSarsa(monteCarloInterface, sarsaEvaluation(monteCarloInterface), lambda, featureMapper, learningRate, w);
  }

  public final DoubleTrueOnlineSarsa doubleTrueOnline( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, FeatureMapper featureMapper, //
      LearningRate learningRate1, LearningRate learningRate2, FeatureWeight w1, FeatureWeight w2) {
    return new DoubleTrueOnlineSarsa(monteCarloInterface, sarsaEvaluation(monteCarloInterface), lambda, featureMapper, learningRate1, learningRate2, w1, w2);
  }
}
