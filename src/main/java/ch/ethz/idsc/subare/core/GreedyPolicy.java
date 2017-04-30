// code by jph
package ch.ethz.idsc.subare.core;

import java.util.HashMap;
import java.util.Map;

import ch.ethz.idsc.subare.util.FairArgMax;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Extract;

public class GreedyPolicy implements PolicyInterface {
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
  public static GreedyPolicy bestEquiprobable(StandardModel standardModel, Tensor values) {
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

  private final Map<Tensor, Index> map;

  private GreedyPolicy(Map<Tensor, Index> map) {
    this.map = map;
  }

  @Override
  public Scalar policy(Tensor state, Tensor action) {
    Index index = map.get(state);
    return index.containsKey(action) ? RationalScalar.of(1, index.size()) : ZeroScalar.get();
  }

  /** useful for export to Mathematica
   * 
   * @param states
   * @return list of actions optimal for */
  public Tensor flatten(Tensor states) {
    Tensor result = Tensors.empty();
    for (Tensor state : states)
      for (Tensor action : map.get(state).keys())
        result.append(Tensors.of(state, action));
    return result;
  }

  /** print overview of possible actions for given states in console
   * 
   * @param states */
  public void print(Tensor states) {
    System.out.println("greedy:");
    for (Tensor state : states) {
      System.out.println(state + " -> " + map.get(state).keys());
    }
  }
}
