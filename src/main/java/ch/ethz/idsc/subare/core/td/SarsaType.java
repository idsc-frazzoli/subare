// code by jph
package ch.ethz.idsc.subare.core.td;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.LearningRate;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.util.FeatureMapper;
import ch.ethz.idsc.tensor.Scalar;

public enum SarsaType {
  ORIGINAL() {
    @Override
    public Sarsa supply(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
      return new OriginalSarsa(discreteModel, qsa, learningRate);
    }

    @Override
    public TrueOnlineSarsa trueOnline(MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper) {
      return new OriginalTrueOnlineSarsa(monteCarloInterface, lambda, learningRate, featureMapper, null);
    }
  }, //
  EXPECTED() {
    @Override
    public Sarsa supply(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
      return new ExpectedSarsa(discreteModel, qsa, learningRate);
    }

    @Override
    public TrueOnlineSarsa trueOnline(MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper) {
      return new ExpectedTrueOnlineSarsa(monteCarloInterface, lambda, learningRate, featureMapper, null);
    }
  }, //
  QLEARNING() {
    @Override
    public Sarsa supply(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
      return new QLearning(discreteModel, qsa, learningRate);
    }

    @Override
    public TrueOnlineSarsa trueOnline(MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper) {
      return new QLearningTrueOnlineSarsa(monteCarloInterface, lambda, learningRate, featureMapper, null);
    }
  }, //
  ;
  // ---
  // TODO JAN rename function
  public abstract Sarsa supply(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate);

  public abstract TrueOnlineSarsa trueOnline( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper);
}
