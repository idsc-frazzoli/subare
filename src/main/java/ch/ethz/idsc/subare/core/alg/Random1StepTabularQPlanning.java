// code by jph
package ch.ethz.idsc.subare.core.alg;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.SampleModel;
import ch.ethz.idsc.subare.core.td.QLearning;
import ch.ethz.idsc.subare.core.util.DiscreteQsa;
import ch.ethz.idsc.subare.core.util.DiscreteUtils;
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
public class Random1StepTabularQPlanning {
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

  public void setUpdateFactor(Scalar alpha) {
    this.alpha = alpha;
  }

  public void batch() {
    Index index = DiscreteUtils.build(discreteModel, discreteModel.states());
    List<Tensor> list = index.keys().flatten(0).collect(Collectors.toList());
    Collections.shuffle(list);
    for (Tensor key : list)
      step(key.get(0), key.get(1));
  }

  public void step(Tensor state0, Tensor action0) {
    Tensor state1 = sampleModel.move(state0, action0);
    Scalar reward = sampleModel.reward(state0, action0, state1);
    Scalar max = discreteModel.actions(state1).flatten(0) //
        .map(action1 -> qsa.value(state1, action1)) //
        .reduce(Max::of).get();
    Scalar value0 = qsa.value(state0, action0);
    // TODO alpha should decrease over time
    Scalar delta = reward.add(gamma.multiply(max)).subtract(value0).multiply(alpha);
    qsa.assign(state0, action0, value0.add(delta));
  }
}
