// Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
const gpu = new GPU();
const pi = 3.141592653589793
const boundary = [50, 50, 50] // (x, y, z), y pointing up
const cell_size = 2.51
const cell_recpr = 1.0 / cell_size

Array.prototype.norm = function () {
    var s = 0;
    for (var i = 0; i < this.length; i++) {
        s += this[i] * this[i];
    }
    return Math.sqrt(s);
}

function round_up(f, s) {
    return (Math.floor(f * cell_recpr / s) + 1) * s
}

grid_size = [round_up(boundary[0], 1), round_up(boundary[1], 1), round_up(boundary[2], 1)]
console.log(grid_size)

const dim = 3
const num_particles_x = 20
const num_particles_y = 20
const num_particles_z = 20
const num_particles = num_particles_x * num_particles_y * num_particles_z
const max_num_particles_per_cell = 100
const max_num_neighbors = 100
const time_delta = 1.0 / 20.0
const epsilon = 1e-5
const particle_radius_in_world = 0.3

// PBF params
const h = 1.1
const mass = 1.0
const rho0 = 1.0
const lambda_epsilon = 100.0
const pbf_num_iters = 5
const corr_deltaQ_coeff = 0.3
const corrK = 0.001
// Need ti.pow()
// corrN = 4.0
const neighbor_radius = h * 1.05

const poly6_factor = 315.0 / 64.0 / pi
const spiky_grad_factor = -45.0 / pi

var old_positions
var positions
var velocities
var lambdas
var position_deltas
// 0: x-pos, 1: timestep in sin()
var board_states


var grid_num_particles
var grid2particles
var particle_num_neighbors
var particle_neighbors

function norm(a, b, dim) {
    var s = 0;
    for (var i = 0; i < dim; i++) {
        s += Math.pow(a[i] - b[i], 2)
    }
    s = Math.sqrt(s);
    return s;
}

function createVariables() {
    positions = []
    old_positions = []
    velocities = []
    position_deltas = []
    lambdas = []
    for (var i = 0; i < num_particles; i++) {
        lambdas.push([]);
        positions.push([[], [], []])
        old_positions.push([[], [], []])
        velocities.push([[], [], []])
        position_deltas.push([[], [], []])
    }

    board_states = [[], [], []];

    grid_num_particles = [] // x
    for (var i = 0; i < grid_size[0]; i++) {
        grid_num_particles.push([])
        for (var j = 0; j < grid_size[1]; j++) {
            grid_num_particles[i].push([])
            for (var k = 0; k < grid_size[2]; k++) {
                grid_num_particles[i][j].push([]);
            }
        }
    }

    grid2particles = [] // x
    for (var i = 0; i < grid_size[0]; i++) {
        grid2particles.push([]) // y
        for (var j = 0; j < grid_size[1]; j++) {
            grid2particles[i].push([]) // z;
            for (var k = 0; k < grid_size[2]; k++) {
                grid2particles[i][j].push([]) // hash
                for (var l = 0; l < max_num_particles_per_cell; l++) {
                    grid2particles[i][j][k].push([])
                }
            }
        }
    }

    particle_num_neighbors = []
    particle_neighbors = []
    for (var i = 0; i < num_particles; i++) {
        particle_num_neighbors.push([]);
        particle_neighbors.push([]);
        for (var j = 0; j < max_num_neighbors; j++) {
            particle_neighbors[i].push(-1)
        }
    }

}


function poly6_value(s, h) {
    result = 0.0
    if (0 < s && s < h) {
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    }
    return result
}


function spiky_gradient(r, h) {
    result = [0.0, 0.0, 0.0]
    r_len = r.norm()
    if (0 < r_len && r_len < h) {
        var x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result[0] = r[0] * g_factor / r_len
        result[1] = r[1] * g_factor / r_len
        result[2] = r[2] * g_factor / r_len
    }
    return result
}

function compute_scorr(pos_ji) {
    // Eq (13)
    x = poly6_value(pos_ji.norm(), h) / poly6_value(corr_deltaQ_coeff * h, h)
    // pow(x, 4)
    x = x * x
    x = x * x
    return (-corrK) * x
}

function get_cell(pos) {
    return [Math.floor(pos[0] * cell_recpr), Math.floor(pos[1] * cell_recpr), Math.floor(pos[2] * cell_recpr)]
}


function is_in_grid(c) {
    // @c: Vector(i32)
    return 0 <= c[0] && c[0] < grid_size[0] && 0 <= c[1] && c[1] < grid_size[1] && 0 <= c[2] && c[2] < grid_size[2]
}

function confine_position_to_boundary(p) {
    bmin = particle_radius_in_world
    bmax = boundary[0] - particle_radius_in_world
    //
    ret = [0, 0, 0]
    for (var i = 0; i < dim; i++) {
        if (p[i] <= bmin)
            ret[i] = bmin
        else if (p[i] >= bmax)
            ret[i] = bmax
        else
            ret[i] = p[i]
    }
    return ret
}

const settings = {
    output: { x: num_particles },
    constants: {
        time_delta: time_delta,
        particle_radius_in_world: particle_radius_in_world,

    },
};
const kernelUpdate = gpu.createKernel(
    function (positions, velocities) {
        function confine_position_to_boundary(p) {
            var bmin = this.constants.particle_radius_in_world
            var bmax = 50 - this.constants.particle_radius_in_world
            //
            var ret = [0, 0, 0]
            for (var i = 0; i < 3; i++) {
                if (p[i] <= bmin)
                    ret[i] = bmin
                else if (p[i] >= bmax)
                    ret[i] = bmax
                else
                    ret[i] = p[i]
            }
            return ret
        }
        var pos = [0, 0, 0]
        pos[0] = positions[this.thread.x][0] + this.constants.time_delta * velocities[this.thread.x][0]
        pos[1] = positions[this.thread.x][1] + this.constants.time_delta * (velocities[this.thread.x][1] - this.constants.time_delta * 9.8)
        pos[2] = positions[this.thread.x][2] + this.constants.time_delta * velocities[this.thread.x][2]
        pos = confine_position_to_boundary(pos);
        return pos
    }, settings)

const kernelGrid = gpu.createKernel(
    function () {
        return 0
    }).setOutput(grid_size)
function prologue() {
    // save old positions
    for (var p_i = 0; p_i < num_particles; p_i++) {
        for (var j = 0; j < dim; j++) {
            old_positions[p_i][j] = positions[p_i][j];
        }
    }
    // apply gravity within boundary

    positions = kernelUpdate(positions, velocities)
    // clear neighbor lookup table

    grid_num_particles = kernelGrid()
    // const kernelPart = gpu.createKernel(
    //     function() {
    //         return -1
    //     }).setOutput([num_particles, max_num_neighbors])
    //particle_neighbors = kernelPart();
    for (var p_i = 0; p_i < num_particles; p_i++)
        for (var j = 0; j < max_num_neighbors; j++)
            particle_neighbors[p_i][j] = -1;

    // update grid
    for (var p_i = 0; p_i < num_particles; p_i++) {
        c = get_cell(positions[p_i])
        offs = grid_num_particles[c[0]][c[1]][c[2]]
        if (offs >= max_num_particles_per_cell - 1) continue;
        grid_num_particles[c[0]][c[1]][c[2]] += 1;
        grid2particles[c[0]][c[1]][c[2]][offs] = p_i;
    }

    for (var p_i = 0; p_i < num_particles; p_i++) {
        pos_i = [0, 0, 0]
        pos_i[0] = positions[p_i][0]
        pos_i[1] = positions[p_i][1]
        pos_i[2] = positions[p_i][2]
        cell = get_cell(pos_i)
        nb_i = 0
        for (var offsX = -1; offsX < 2; offsX++) {
            for (var offsY = -1; offsY < 2; offsY++) {
                for (var offsZ = -1; offsZ < 2; offsZ++) {
                    cell_to_check = [offsX + cell[0], offsY + cell[1], offsZ + cell[2]]
                    if (is_in_grid(cell_to_check)) {
                        for (var j = 0; j < grid_num_particles[cell_to_check[0]][cell_to_check[1]][cell_to_check[2]]; j++) {
                            p_j = grid2particles[cell_to_check[0]][cell_to_check[1]][cell_to_check[2]][j];
                            if (nb_i < max_num_neighbors && p_j != p_i &&
                                norm(pos_i, positions[p_j], dim) < neighbor_radius) {
                                particle_neighbors[p_i][nb_i] = p_j
                                nb_i += 1
                            }
                        }
                    }
                }
            }
        }
        particle_num_neighbors[p_i] = nb_i
    }
    // find particle neighbors
}
const settings0 = {
    output: { x: num_particles },
    constants: {
        poly6_factor: 315.0 / 64.0 / pi,
        spiky_grad_factor: -45.0 / pi,
        h: 1.1,
        mass: 1.0,
        rho0: 1.0,
        lambda_epsilon: 100.0,
        corr_deltaQ_coeff: 0.3,
        corrK: 0.001,
    },
}

const kernelApplyDelta = gpu.createKernel(
    function (positions, position_deltas) {
        return [positions[this.thread.x][0] + position_deltas[this.thread.x][0],
        positions[this.thread.x][1] + position_deltas[this.thread.x][1],
        positions[this.thread.x][2] + position_deltas[this.thread.x][2]]
    }
).setOutput([num_particles])

const settings1 = {
    output: { x: num_particles },
    constants: {
        poly6_factor: 315.0 / 64.0 / pi,
        spiky_grad_factor: -45.0 / pi,
        h: 1.1,
        mass: 1.0,
        rho0: 1.0,
        lambda_epsilon: 100.0,
        corr_deltaQ_coeff: 0.3,
        corrK: 0.001,
    },
}

const kernelComputePosDelta = gpu.createKernel(
    function (positions, lambdas, particle_num_neighbors, particle_neighbors) {
        function poly6_value(s) {
            var h = this.constants.h
            var result = 0.0
            if (0 < s && s < h) {
                var x = (h * h - s * s) / (h * h * h)
                result = this.constants.poly6_factor * x * x * x
            }
            return result
        }
        function spiky_gradient(r) {
            var result = [0.0, 0.0, 0.0]
            var r_len = Math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
            if (0 < r_len && r_len < this.constants.h) {
                var x = (this.constants.h - r_len) / (this.constants.h * this.constants.h * this.constants.h)
                var g_factor = this.constants.spiky_grad_factor * x * x
                result[0] = r[0] * g_factor / r_len
                result[1] = r[1] * g_factor / r_len
                result[2] = r[2] * g_factor / r_len
            }
            return result
        }
        function compute_scorr(pos_ji) {
            // Eq (13)
            var n = Math.sqrt(pos_ji[0] * pos_ji[0] + pos_ji[1] * pos_ji[1] + pos_ji[2] * pos_ji[2])
            var x = poly6_value(n) / poly6_value(this.constants.corr_deltaQ_coeff * this.constants.h)
            // pow(x, 4)
            return (-this.constants.corrK) * Math.pow(x, 4)
        }
        var pos_i = positions[this.thread.x]
        var lambda_i = lambdas[this.thread.x]

        var pos_delta_i = [0.0, 0.0, 0.0]
        for (var j = 0; j < particle_num_neighbors[this.thread.x]; j++) {
            var p_j = particle_neighbors[this.thread.x][j]
            if (p_j < 0)
                break;
            var lambda_j = lambdas[p_j]
            var pos_ji = [0, 0, 0]
            pos_ji[0] = positions[this.thread.x][0] - positions[p_j][0]
            pos_ji[1] = positions[this.thread.x][1] - positions[p_j][1]
            pos_ji[2] = positions[this.thread.x][2] - positions[p_j][2]
            var scorr_ij = compute_scorr(pos_ji)
            var grad = spiky_gradient(pos_ji);
            pos_delta_i[0] += (lambda_i + lambda_j + scorr_ij) * grad[0]
            pos_delta_i[1] += (lambda_i + lambda_j + scorr_ij) * grad[1]
            pos_delta_i[2] += (lambda_i + lambda_j + scorr_ij) * grad[2]
        }

        pos_delta_i[0] /= this.constants.rho0
        pos_delta_i[1] /= this.constants.rho0
        pos_delta_i[2] /= this.constants.rho0

        var ret = [pos_delta_i[0], pos_delta_i[1], pos_delta_i[2]]
        return ret;
    }, settings1).setPipeline(true)

const kernelComputeLambda = gpu.createKernel(
    function (positions, particle_neighbors, particle_num_neighbors) {
        function poly6_value(s) {
            if (0 < s && s < this.constants.h) {
                var x = (this.constants.h * this.constants.h - s * s) / Math.pow(this.constants.h, 3)
                return this.constants.poly6_factor * x * x * x
            }
            return 0
        }
        function spiky_gradient(r) {
            var r_len = Math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
            if (0 < r_len && r_len < this.constants.h) {
                var x = (this.constants.h - r_len) / (this.constants.h * this.constants.h * this.constants.h)
                var g_factor = this.constants.spiky_grad_factor * x * x
                return [r[0] * g_factor / r_len, r[1] * g_factor / r_len, r[2] * g_factor / r_len]
            }
            return [0, 0, 0]
        }
        var grad_i = [0.0, 0.0, 0.0]
        var sum_gradient_sqr = 0.0
        var density_constraint = 0.0
        for (var j = 0; j < particle_num_neighbors[this.thread.x]; j++) {
            var p_j = particle_neighbors[this.thread.x][j]
            var r = [positions[this.thread.x][0] - positions[p_j][0], positions[this.thread.x][1] - positions[p_j][1], positions[this.thread.x][2] - positions[p_j][2]]
            var grad_j = spiky_gradient(r)
            grad_i[0] += grad_j[0]
            grad_i[1] += grad_j[1]
            grad_i[2] += grad_j[2]
            sum_gradient_sqr += grad_j[0] * grad_j[0] + grad_j[1] * grad_j[1] + grad_j[2] * grad_j[2];
            density_constraint += poly6_value(Math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]))
        }

        density_constraint = (this.constants.mass * density_constraint / this.constants.rho0) - 1.0

        sum_gradient_sqr += grad_i[0] * grad_i[0] + grad_i[1] * grad_i[1] + grad_i[2] * grad_i[2];
        return (-density_constraint) / (sum_gradient_sqr + this.constants.lambda_epsilon)
    }, settings0).setPipeline(true)


function substep() {
    lambdas = kernelComputeLambda(positions, particle_neighbors, particle_num_neighbors);
    // compute position deltas
    position_deltas = kernelComputePosDelta(positions, lambdas, particle_num_neighbors, particle_neighbors)
    // apply position deltas
    positions = kernelApplyDelta(positions, position_deltas.toArray())
}

function epilogue() {
    // confine to boundary
    for (var i = 0; i < num_particles; i++) {
        cptb = confine_position_to_boundary(positions[i])
        positions[i][0] = cptb[0]
        positions[i][1] = cptb[1]
        positions[i][2] = cptb[2]
    }
    // update velocities
    for (var i = 0; i < num_particles; i++) {
        velocities[i][0] = (positions[i][0] - old_positions[i][0]) / time_delta
        velocities[i][1] = (positions[i][1] - old_positions[i][1]) / time_delta
        velocities[i][2] = (positions[i][2] - old_positions[i][2]) / time_delta
    }
    // no vorticity/xsph because we cannot do cross product in 2D...

}


function run_pbf() {
    var d = new Date();
    var n = d.getTime();
    prologue()
    var d = new Date();
    console.log("Prologue: " + (d.getTime() - n))
    n = d.getTime();
    for (var i = 0; i < pbf_num_iters; i++) {
        substep()
    }
    var d = new Date();
    console.log("Substep: " + (d.getTime() - n))
    n = d.getTime();
    epilogue()
    var d = new Date();
    console.log("Epilogue: " + (d.getTime() - n))
    n = d.getTime();
}


function init_particles() {
    for (var i = 0; i < num_particles; i++) {
        delta = h * 0.8
        offs = [(boundary[0] - delta * num_particles_x), 0, 0]
        var x = Math.floor(i / num_particles_y) % num_particles_x;
        var y = (i % num_particles_y);
        var z = Math.floor(Math.floor(i / num_particles_x) / num_particles_y)
        positions[i][0] = x * delta + offs[0]
        positions[i][1] = y * delta + offs[1]
        positions[i][2] = z * delta + offs[2]
        for (var c = 0; c < dim; c++)
            velocities[i][c] = 0
    }
    //board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])
}


function print_stats() {
    console.log('PBF stats:')
    max = 0;
    avg = 0;
    for (var i = 0; i < grid_size[0]; i++)
        for (var j = 0; j < grid_size[1]; j++)
            for (var k = 0; k < grid_size[2]; k++) {
                if (grid_num_particles[i][j][k] > max)
                    max = grid_num_particles[i][j][k];
                avg += grid_num_particles[i][j][k];
            }
    avg /= (grid_size[0] * grid_size[1] * grid_size[2])
    console.log("  #particles per cell: avg={" + avg + "} max={" + max + "}")
    max = 0;
    avg = 0;
    for (var i = 0; i < num_particles; i++) {
        avg += particle_num_neighbors[i];
        if (particle_num_neighbors[i] > max)
            max = particle_num_neighbors[i]
    }
    avg
    console.log("  #neighbors per particle: sum={" + avg + "} max={" + max + "}")
    console.log(positions[0])
}


function main() {
    createVariables()
    init_particles()
    frame = 0;
    while (true) {
        frame++;
        run_pbf()
        if (frame % 20 == 1)
            print_stats()
        if (frame > 1000)
            break;
    }
}

