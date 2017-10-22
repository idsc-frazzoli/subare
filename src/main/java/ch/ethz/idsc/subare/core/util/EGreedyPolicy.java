// code by jph
package ch.ethz.idsc.subare.core.util;

import java.util.Map;
import java.util.Objects;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.core.QsaInterface;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RationalScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;

/** p.33 */
public class EGreedyPolicy implements Policy {
  public static Policy bestEquiprobable(DiscreteModel discreteModel, QsaInterface qsa, Scalar epsilon) {
    EGreedyPolicyBuilder builder = new EGreedyPolicyBuilder(discreteModel, qsa);
    discreteModel.states().forEach(builder::append);
    return new EGreedyPolicy(builder.map, epsilon, builder.sizes);
  }

  public static Policy bestEquiprobable(DiscreteModel discreteModel, QsaInterface qsa, Scalar epsilon, Tensor state) {
    EGreedyPolicyBuilder builder = new EGreedyPolicyBuilder(discreteModel, qsa);
    builder.append(state);
    return new EGreedyPolicy(builder.map, epsilon, builder.sizes);
  }

  private final Map<Tensor, Index> map;
  /** probability of choosing a non-optimal action, if there is at least one non-optimal action */
  private final Scalar epsilon;
  private final Map<Tensor, Integer> sizes;

  EGreedyPolicy(Map<Tensor, Index> map, Scalar epsilon, Map<Tensor, Integer> sizes) {
    this.map = map;
    this.epsilon = epsilon;
    this.sizes = sizes;
    if (Objects.isNull(sizes) && Scalars.nonZero(epsilon))
      throw new RuntimeException("sizes invalid for " + epsilon);
  }

  @Override
  public Scalar probability(Tensor state, Tensor action) {
    Index index = map.get(state);
    final int optimalCount = index.size();
    if (Objects.isNull(sizes)) // greedy
      return index.containsKey(action) ? RationalScalar.of(1, optimalCount) : RealScalar.ZERO;
    // ---
    final int nonOptimalCount = sizes.get(state) - optimalCount;
    if (nonOptimalCount == 0) // no non-optimal action exists
      return RationalScalar.of(1, optimalCount);
    if (index.containsKey(action))
      return RealScalar.ONE.subtract(epsilon).divide(RealScalar.of(optimalCount));
    return epsilon.divide(RealScalar.of(nonOptimalCount));
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
    for (Tensor state : states)
      System.out.println(state + " -> " + map.get(state).keys());
  }
}
