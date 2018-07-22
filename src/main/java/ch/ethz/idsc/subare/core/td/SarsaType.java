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
    public Sarsa supply(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
      return new Sarsa(discreteModel, qsa, learningRate, SarsaEvaluationType.ORIGINAL);
    }

    @Override
    public TrueOnlineSarsa trueOnline(MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, Tensor w,
        FeatureMapper featureMapper) {
      return new TrueOnlineSarsa(monteCarloInterface, SarsaEvaluationType.ORIGINAL, lambda, learningRate, featureMapper, w);
    }
  }, //
  EXPECTED() {
    @Override
    public Sarsa supply(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
      return new Sarsa(discreteModel, qsa, learningRate, SarsaEvaluationType.EXPECTED);
    }

    @Override
    public TrueOnlineSarsa trueOnline(MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, Tensor w,
        FeatureMapper featureMapper) {
      return new TrueOnlineSarsa(monteCarloInterface, SarsaEvaluationType.EXPECTED, lambda, learningRate, featureMapper, w);
    }
  }, //
  QLEARNING() {
    @Override
    public Sarsa supply(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate) {
      return new Sarsa(discreteModel, qsa, learningRate, SarsaEvaluationType.QLEARNING);
    }

    @Override
    public TrueOnlineSarsa trueOnline(MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, Tensor w,
        FeatureMapper featureMapper) {
      return new TrueOnlineSarsa(monteCarloInterface, SarsaEvaluationType.QLEARNING, lambda, learningRate, featureMapper, w);
    }
  }, //
  ;
  // ---
  // TODO JAN rename function
  public abstract Sarsa supply(DiscreteModel discreteModel, QsaInterface qsa, LearningRate learningRate);

  public Sarsa supply(DiscreteModel discreteModel, LearningRate learningRate) {
    return supply(discreteModel, null, learningRate);
  }

  public abstract TrueOnlineSarsa trueOnline( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, Tensor w, FeatureMapper featureMapper);

  public TrueOnlineSarsa trueOnline( //
      MonteCarloInterface monteCarloInterface, Scalar lambda, LearningRate learningRate, FeatureMapper featureMapper) {
    return trueOnline(monteCarloInterface, lambda, learningRate, null, featureMapper);
  }
}
