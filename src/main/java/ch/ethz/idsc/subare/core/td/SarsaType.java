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
  public final Sarsa supply(DiscreteModel discreteModel, LearningRate learningRate) {
    return supply(discreteModel, null, learningRate);
  }

  public final Sarsa supply(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
    return new Sarsa(sarsaEvaluation(discreteModel), discreteModel, qsa, learningRate);
  }

  public final DoubleSarsa doubleSarsa(DiscreteModel discreteModel, QsaInterface qsa1, QsaInterface qsa2, LearningRate learningRate1,
      LearningRate learningRate2) {
    return new DoubleSarsa(sarsaEvaluation(discreteModel), discreteModel, qsa1, qsa2, learningRate1, learningRate2);
  }

  public final TrueOnlineSarsa trueOnline( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper) {
    return trueOnline(monteCarloInterface, lambda, learningRate, null, featureMapper);
  }

  public final TrueOnlineSarsa trueOnline( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, Tensor w, FeatureMapper featureMapper) {
    return new TrueOnlineSarsa(monteCarloInterface, sarsaEvaluation(monteCarloInterface), lambda, learningRate, featureMapper, w);
  }

  public final DoubleTrueOnlineSarsa doubleTrueOnline( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate1, LearningRate learningRate2, FeatureMapper featureMapper) {
    return new DoubleTrueOnlineSarsa(monteCarloInterface, sarsaEvaluation(monteCarloInterface), lambda, learningRate1, learningRate2, featureMapper);
  }
}
