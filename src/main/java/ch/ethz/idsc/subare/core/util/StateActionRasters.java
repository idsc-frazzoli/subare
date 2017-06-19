// code by jph
package ch.ethz.idsc.subare.core.util;

import java.awt.Dimension;
import java.awt.Point;
import java.awt.image.BufferedImage;
import java.util.List;

import ch.ethz.idsc.subare.core.DiscreteModel;
import ch.ethz.idsc.subare.core.Policy;
import ch.ethz.idsc.subare.util.Colorscheme;
import ch.ethz.idsc.subare.util.ImageResize;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.io.ImageFormat;
import ch.ethz.idsc.tensor.opt.Interpolation;
import ch.ethz.idsc.tensor.sca.Clip;

public enum StateActionRasters {
  ;
  private static final Interpolation COLORSCHEME = Colorscheme.classic();
  private static final Tensor BASE = Tensors.vector(255);

  /** @param stateActionRaster
   * @param qsa scaled to contain values in the interval [0, 1]
   * @return */
  public static Tensor render(StateActionRaster stateActionRaster, DiscreteQsa qsa) {
    DiscreteModel discreteModel = stateActionRaster.discreteModel();
    Dimension dimension = stateActionRaster.dimension();
    final Tensor tensor = Array.zeros(dimension.width, dimension.height, 4);
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state)) {
        Point point = stateActionRaster.point(state, action);
        if (point != null) {
          Scalar sca = qsa.value(state, action);
          tensor.set(COLORSCHEME.get(BASE.multiply(sca)), point.x, point.y);
        }
      }
    return tensor;
  }

  public static Tensor render(StateActionRaster stateActionRaster, Policy policy) {
    return render(stateActionRaster, Policies.toQsa(stateActionRaster.discreteModel(), policy));
  }

  public static Tensor qsaPolicy(StateActionRaster stateActionRaster, DiscreteQsa qsa) {
    Tensor image1 = render(stateActionRaster, DiscreteValueFunctions.rescaled(qsa));
    Policy policy = GreedyPolicy.bestEquiprobable(stateActionRaster.discreteModel(), qsa);
    Tensor image2 = render(stateActionRaster, policy);
    List<Integer> list = Dimensions.of(image1);
    list.set(0, 3);
    return Join.of(0, image1, Array.zeros(list), image2);
  }

  public static Tensor qsaPolicyRef(StateActionRaster stateActionRaster, DiscreteQsa qsa, DiscreteQsa ref) {
    Tensor image1 = render(stateActionRaster, DiscreteValueFunctions.rescaled(qsa));
    Policy policy = GreedyPolicy.bestEquiprobable(stateActionRaster.discreteModel(), qsa);
    Tensor image2 = render(stateActionRaster, policy);
    Scalar qdelta = stateActionRaster.scaleQdelta();
    Tensor image3 = render(stateActionRaster, DiscreteValueFunctions.logisticDifference(qsa, ref, qdelta));
    List<Integer> list = Dimensions.of(image1);
    list.set(0, 3);
    return ImageResize.of( //
        Join.of(0, image1, Array.zeros(list), image2, Array.zeros(list), image3), stateActionRaster.magify());
  }

  // only called in ch04.gridworld
  public static BufferedImage qsaLossRef(StateActionRaster stateActionRaster, DiscreteQsa qsa, DiscreteQsa ref) {
    Tensor image1 = render(stateActionRaster, DiscreteValueFunctions.rescaled(qsa));
    Scalar scale = stateActionRaster.scaleLoss();
    DiscreteQsa loss = Loss.asQsa(stateActionRaster.discreteModel(), ref, qsa);
    loss = loss.create(loss.values().flatten(0) //
        .map(tensor -> tensor.multiply(scale)) //
        .map(Clip.UNIT::of));
    Tensor image2 = render(stateActionRaster, loss);
    Tensor image3 = render(stateActionRaster, DiscreteValueFunctions.logisticDifference(qsa, ref));
    List<Integer> list = Dimensions.of(image1);
    int dim = stateActionRaster.joinAlongDimension();
    list.set(dim, 1);
    return ImageFormat.of(ImageResize.of( //
        Join.of(dim, image1, Array.zeros(list), image2, Array.zeros(list), image3), stateActionRaster.magify()));
  }
}
