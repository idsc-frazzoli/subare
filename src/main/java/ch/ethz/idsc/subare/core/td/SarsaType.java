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
      return OriginalTrueOnlineSarsa.of(monteCarloInterface, lambda, learningRate, featureMapper);
    }
  }, //
  EXPECTED() {
    @Override
    public Sarsa supply(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
      return new ExpectedSarsa(discreteModel, qsa, learningRate);
    }

    @Override
    public TrueOnlineSarsa trueOnline(MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper) {
      return ExpectedTrueOnlineSarsa.of(monteCarloInterface, lambda, learningRate, featureMapper);
    }
  }, //
  QLEARNING() {
    @Override
    public Sarsa supply(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
      return new QLearning(discreteModel, qsa, learningRate);
    }

    @Override
    public TrueOnlineSarsa trueOnline(MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper) {
      return QLearningTrueOnlineSarsa.of(monteCarloInterface, lambda, learningRate, featureMapper);
    }
  }, //
  ;
  // ---
  // TODO JAN rename function
  public abstract Sarsa supply(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate);

  public abstract TrueOnlineSarsa trueOnline( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper);
}
