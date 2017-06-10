// code by jph
package ch.ethz.idsc.subare.core.alg;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.SampleModel;
import ch.ethz.idsc.subare.core.StepDigest;
import ch.ethz.idsc.subare.core.StepInterface;
import ch.ethz.idsc.subare.core.td.QLearning;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
import ch.ethz.idsc.subare.core.util.StepAdapter;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.red.Max;

/** similar to {@link QLearning} except does not use episodes but single steps
 * 
 * similar to {@link ActionValueIteration} but with gauss-seidel updates
 * therefore not parallel()
 * 
 * see box on p.169
 * 
 * algorithm performs poorly when rewards are unevenly distributed */
public class Random1StepTabularQPlanning implements StepDigest {
  private final DiscreteModel discreteModel;
  private final SampleModel sampleModel;
  private final DiscreteQsa qsa;
  private final Scalar gamma;
  private Scalar alpha = null;

  public Random1StepTabularQPlanning( //
      DiscreteModel discreteModel, SampleModel sampleModel, QsaInterface qsa) {
    this.discreteModel = discreteModel;
    this.sampleModel = sampleModel;
    this.qsa = (DiscreteQsa) qsa;
    this.gamma = discreteModel.gamma();
  }

  /** @param alpha should decrease over time */
  public void setLearningRate(Scalar alpha) {
    this.alpha = alpha;
  }

  public void batch() {
    Index index = DiscreteUtils.build(discreteModel, discreteModel.states());
    List<Tensor> list = index.keys().flatten(0).collect(Collectors.toList());
    Collections.shuffle(list);
    for (Tensor key : list)
      step(key.get(0), key.get(1));
  }

  public void step(Tensor state, Tensor action) {
    Tensor next = sampleModel.move(state, action);
    Scalar reward = sampleModel.reward(state, action, next);
    digest(new StepAdapter(state, action, reward, next));
  }

  @Override
  public void digest(StepInterface stepInterface) {
    Tensor state0 = stepInterface.prevState();
    Tensor action = stepInterface.action();
    Scalar reward = stepInterface.reward();
    Tensor state1 = stepInterface.nextState();
    // ---
    Scalar max = discreteModel.actions(state1).flatten(0) //
        .map(action1 -> qsa.value(state1, action1)) //
        .reduce(Max::of).get();
    Scalar value0 = qsa.value(state0, action);
    Scalar delta = reward.add(gamma.multiply(max)).subtract(value0).multiply(alpha);
    qsa.assign(state0, action, value0.add(delta));
  }
}
