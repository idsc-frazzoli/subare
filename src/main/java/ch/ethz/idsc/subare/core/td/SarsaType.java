// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;

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

  public final DoubleSarsa doubleSarsa(DiscreteModel discreteModel, LearningRate learningRate1, LearningRate learningRate2, QsaInterface qsa1,
      QsaInterface qsa2) {
    return new DoubleSarsa(sarsaEvaluation(discreteModel), discreteModel, learningRate1, learningRate2, qsa1, qsa2);
  }

  public final TrueOnlineSarsa trueOnline( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, FeatureMapper featureMapper, LearningRate learningRate, Tensor w) {
    return new TrueOnlineSarsa(monteCarloInterface, sarsaEvaluation(monteCarloInterface), lambda, featureMapper, learningRate, w);
  }

  public final DoubleTrueOnlineSarsa doubleTrueOnline( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, FeatureMapper featureMapper, LearningRate learningRate1, LearningRate learningRate2, Tensor w1,
      Tensor w2) {
    return new DoubleTrueOnlineSarsa(monteCarloInterface, sarsaEvaluation(monteCarloInterface), lambda, featureMapper, learningRate1, learningRate2, w1, w2);
  }
}
