// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.util.FairArgMax;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Extract;

public class GreedyPolicy extends EGreedyPolicy {
  /** exact implementation of equiprobable greedy policy:
   * if two or more states s1,s2, ... have equal value
   * v(s1)==v(s2)
   * then they are assigned equal probability
   * 
   * in case there is no unique maximum value
   * there are infinitely many greedy policies
   * and not a unique one policy.
   * 
   * @param standardModel
   * @param values of standardModel.states()
   * @return */
  public static GreedyPolicy bestEquiprobableGreedy(StandardModel standardModel, Tensor values) {
    Map<Tensor, Index> map = new HashMap<>();
    for (Tensor state : standardModel.states()) {
      Tensor actions = standardModel.actions(state);
      Tensor va = Tensor.of(actions.flatten(0) //
          .map(action -> standardModel.qsa(state, action, values)));
      FairArgMax fairArgMax = FairArgMax.of(va);
      Tensor feasible = Extract.of(actions, fairArgMax.options());
      map.put(state, Index.build(feasible));
    }
    return new GreedyPolicy(map);
  }

  // this simplicity may be the reason why q(s,a) is preferred over v(s)
  public static GreedyPolicy bestEquiprobableGreedy(DiscreteModel discreteModel, QsaInterface qsa) {
    Map<Tensor, Index> map = new HashMap<>();
    for (Tensor state : discreteModel.states()) {
      Tensor actions = discreteModel.actions(state);
      Tensor va = actions.map(action -> qsa.value(state, action));
      FairArgMax fairArgMax = FairArgMax.of(va);
      Tensor feasible = Extract.of(actions, fairArgMax.options());
      map.put(state, Index.build(feasible));
    }
    return new GreedyPolicy(map);
  }

  GreedyPolicy(Map<Tensor, Index> map) {
    super(map, ZeroScalar.get(), null);
  }
}
